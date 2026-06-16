"""Approach 3 — Agentic RAG (§4).

The LLM is handed tools and runs a retrieve–assess–retrieve loop until it can
answer or concludes it cannot. The tools mirror what an HR specialist actually does
— keyword-search, scan the TOC, read a section, double-check a quote — which keeps
the agent's trace legible to a human reviewer.

verify_quote is exposed as a tool AND re-run server-side: letting the agent check
its own quotes cuts retries; re-verifying outside the loop closes the self-grading
hole. A hard step cap bounds cost and prevents loops.
"""
from .contract import Answer
from .generate import build_claims
from .retrieve import hybrid_search
from .tree import render_toc
from .verify import verify_answer, verify_quote

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["source_id", "quote"],
                        },
                    },
                },
                "required": ["text", "citations"],
            },
        }
    },
    "required": ["claims"],
}

TOOLS = [
    {"name": "search_handbook",
     "description": "Hybrid keyword+semantic search. Returns top chunks with ids, section paths, pages.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "browse_toc",
     "description": "Return the handbook's table of contents with node ids and summaries.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "read_section",
     "description": "Return full text of a TOC node by node_id.",
     "parameters": {"type": "object",
                    "properties": {"node_id": {"type": "string"}},
                    "required": ["node_id"]}},
    {"name": "verify_quote",
     "description": "Check a verbatim quote exists in chunk/node source_id. Use before submitting.",
     "parameters": {"type": "object",
                    "properties": {"quote": {"type": "string"},
                                   "source_id": {"type": "string"}},
                    "required": ["quote", "source_id"]}},
    {"name": "submit_answer",
     "description": "Final answer as claims with citations (chunk/node id + verbatim quote).",
     "parameters": CLAIMS_SCHEMA},
    {"name": "abstain",
     "description": "Use when the handbook does not contain the answer.",
     "parameters": {"type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"]}},
]

AGENT_SYSTEM = """You answer OmniCorp HR policy questions strictly from the Employee Handbook, via tools.
Method: break the question into the distinct policies it touches; search or browse the TOC for each;
read full sections rather than relying on snippets; follow cross-references; verify every quote with
verify_quote before submit_answer. If evidence is missing or contradictory, abstain and say why.
You have at most 8 tool calls."""


class SourceResolver:
    """Resolves a source_id (chunk id or tree node_id) to text/page/section."""

    def __init__(self, store, node_index):
        self.store = store
        self.node_index = node_index

    def get(self, source_id):
        if self.store and source_id in self.store.by_id:
            return self.store.by_id[source_id]
        return self.node_index.get(source_id) if self.node_index else None

    def text(self, source_id) -> str:
        s = self.get(source_id)
        return s.text if s else ""


def run_agent(llm, query: str, store, root, node_index, max_steps: int = 8) -> Answer:
    resolver = SourceResolver(store, node_index)
    msgs = [{"role": "user", "content": query}]
    trace: list[dict] = []

    def exec_tool(name: str, args: dict) -> str:
        if name == "search_handbook":
            ids = hybrid_search(store, args.get("query", ""), k_each=20, k_final=6)
            if not ids:
                return "No matching sections found."
            blocks = []
            for cid in ids:
                c = store.by_id[cid]
                blocks.append(f"[id={cid}] {c.section_path} (pp.{c.page_start}-{c.page_end})\n{c.text}")
            return "\n\n".join(blocks)
        if name == "browse_toc":
            return render_toc(root)
        if name == "read_section":
            node = node_index.get(args.get("node_id", "")) if node_index else None
            return node.text if node else "Unknown node_id."
        if name == "verify_quote":
            ok = verify_quote(args.get("quote", ""), resolver.text(args.get("source_id", "")))
            return "VERIFIED" if ok else "NOT FOUND in that source — re-read it."
        return f"Unknown tool: {name}"

    for _ in range(max_steps):
        resp = llm.chat(messages=msgs, system=AGENT_SYSTEM, tools=TOOLS, max_tokens=2000)
        if not resp.tool_calls:
            break
        msgs.append({"role": "assistant", "content": resp.text,
                     "tool_calls": resp.tool_calls})
        results = []
        for call in resp.tool_calls:
            trace.append({"tool": call.name, "input": call.arguments})
            if call.name == "submit_answer":
                return _finalize(call.arguments, resolver, trace)
            if call.name == "abstain":
                ans = Answer(claims=[], abstained=True)
                ans.retrieval = {"approach": "agentic",
                                 "reason": call.arguments.get("reason", ""), "trace": trace}
                return ans
            results.append({"role": "tool", "tool_call_id": call.id,
                            "content": exec_tool(call.name, call.arguments)})
        msgs.extend(results)

    ans = Answer(claims=[], abstained=True)
    ans.retrieval = {"approach": "agentic", "reason": "step budget exhausted", "trace": trace}
    return ans


def _finalize(args: dict, resolver: SourceResolver, trace: list) -> Answer:
    claims = build_claims(args.get("claims", []), resolver.get)
    answer = Answer(claims=claims, abstained=not claims)
    # server-side re-verification — never trust the agent's own verify_quote calls
    answer = verify_answer(answer, lambda c: resolver.text(c.source_id))
    answer.retrieval = {"approach": "agentic", "trace": trace}
    return answer
