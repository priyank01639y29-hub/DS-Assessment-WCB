"""Top-level orchestration.

Builds the shared layer once (parse → chunk → index → tree), then routes a query to
whichever retrieval approach is requested. Per the guide's recommendation (§5), the
default is PageIndex — a handbook is exactly the TOC-rich document it was designed
for, and section-level citations match how HR already reasons about the document.
"""
import re
import time

from .agent import run_agent
from .chunk import chunk_by_section
from .config import Config, estimate_cost, load_config
from .contract import Answer
from .embeddings import make_embedder
from .llm import make_llm
from .pageindex import answer_pageindex
from .parse import extract_blocks
from .store import InMemoryStore
from .traditional import answer_traditional
from .tree import build_tree, index_nodes

APPROACHES = ("traditional", "pageindex", "agentic")


class PolicyPal:
    def __init__(self, pdf_path: str, config: Config | None = None):
        self.config = config or load_config()
        self.llm = make_llm(self.config)
        self.embedder = make_embedder(self.config)

        self.blocks = extract_blocks(pdf_path)          # §1.1 shared foundation
        self.chunks = chunk_by_section(self.blocks)     # §2.2 traditional/agentic
        self.store = self._make_store()
        self.tree = build_tree(self.blocks)             # §3.2 pageindex/agentic
        self.node_index = index_nodes(self.tree)

    def _make_store(self):
        """Pick the storage backend: in-RAM (NumPy+BM25) or pgvector (DB).

        backend=auto → pgvector when DATABASE_URL is set, else memory; if pgvector is
        chosen by auto-detection but unreachable, fall back to memory with a warning.
        backend=pgvector is strict (errors propagate)."""
        cfg = self.config
        backend = cfg.store_backend
        if backend == "auto":
            backend = "pgvector" if cfg.database_url else "memory"

        if backend == "pgvector":
            from .pgstore import PgVectorStore
            try:
                return PgVectorStore(
                    self.chunks, self.embedder, cfg.database_url,
                    hybrid_method=cfg.hybrid_method, hybrid_alpha=cfg.hybrid_alpha,
                    rebuild=cfg.store_rebuild)
            except Exception as e:
                if cfg.store_backend == "pgvector":      # explicit request → don't hide failures
                    raise
                import sys
                print(f"[policypal] pgvector unavailable ({e}); using in-memory store",
                      file=sys.stderr)

        return InMemoryStore(
            self.chunks, self.embedder,
            k1=cfg.bm25_k1, b=cfg.bm25_b, stem=cfg.keyword_stem,
            hybrid_method=cfg.hybrid_method, hybrid_alpha=cfg.hybrid_alpha)

    # --- the three approaches -------------------------------------------------
    def answer(self, query: str, approach: str = "pageindex") -> Answer:
        before = dict(self.llm.total_usage)
        t0 = time.perf_counter()
        if approach == "traditional":
            ans = answer_traditional(self.llm, self.store, query)
        elif approach == "pageindex":
            ans = answer_pageindex(self.llm, self.tree, query, self.node_index)
        elif approach == "agentic":
            ans = run_agent(self.llm, query, self.store, self.tree, self.node_index)
        else:
            raise ValueError(f"approach must be one of {APPROACHES}, got {approach!r}")
        ans.latency_s = time.perf_counter() - t0
        self._attach_usage(ans, before)
        return ans

    def _attach_usage(self, ans: Answer, before: dict):
        """Per-query token usage = cumulative delta across all LLM calls this answer made."""
        after = self.llm.total_usage
        ans.usage = {
            "input_tokens": after["input_tokens"] - before["input_tokens"],
            "output_tokens": after["output_tokens"] - before["output_tokens"],
            "calls": after["calls"] - before["calls"],
        }
        ans.cost_usd = estimate_cost(ans.usage, self.config)

    def answer_all(self, query: str) -> dict[str, Answer]:
        """Run the query through all three — handy for side-by-side evaluation."""
        return {a: self.answer(query, a) for a in APPROACHES}

    # --- optional complexity router (§5 recommendation 4) ---------------------
    def route(self, query: str) -> str:
        """Cheap heuristic: multi-facet questions → agentic; everything else →
        PageIndex (the default path), with traditional as the recall fallback."""
        multi = bool(re.search(r"\band\b|\bor\b|;|\balso\b|both", query.lower()))
        return "agentic" if multi else "pageindex"

    def answer_routed(self, query: str, approach: str = "auto") -> Answer:
        """Answer with an approach toggle.

        approach="auto" (default): the heuristic router picks (multi-facet →
        agentic, else PageIndex), with a PageIndex→traditional recall fallback.
        Any explicit approach in APPROACHES bypasses the router and runs only
        that one. The approach actually used is recorded in
        ans.retrieval["approach"] — useful after the auto fallback fires."""
        if approach not in ("auto", *APPROACHES):
            raise ValueError(
                f"approach must be 'auto' or one of {APPROACHES}, got {approach!r}")

        if approach != "auto":                       # toggle: user forced one
            ans = self.answer(query, approach)
            ans.retrieval["approach"] = approach
            return ans

        chosen = self.route(query)
        ans = self.answer(query, chosen)
        used = chosen
        # fallback path (§5 rec 3): if PageIndex finds nothing, try hybrid RAG.
        # Both legs actually ran, so the returned answer accounts for both —
        # latency, tokens, and cost all sum across them.
        if chosen == "pageindex" and ans.abstained:
            first = ans
            ans = self.answer(query, "traditional")
            ans.latency_s += first.latency_s
            for k, v in first.usage.items():
                ans.usage[k] = ans.usage.get(k, 0) + v
            ans.cost_usd += first.cost_usd
            used = "traditional (fallback from pageindex)"
        ans.retrieval["approach"] = used
        return ans
