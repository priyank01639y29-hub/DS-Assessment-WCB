# Compare & contrast: six ways to answer questions over your documents with an LLM

When you need an LLM to answer from a body of documents (PolicyPal's HR handbook, say),
the options span a spectrum. They split into **two families**:

- **Retrieve from the source at query time** — the original document stays the source of
  record; you fetch the right piece on demand. *Direct upload, Traditional RAG, PageIndex,
  Agentic RAG.*
- **Compile the source into derived knowledge ahead of time** — you transform the documents
  into a new, durable representation the LLM reasons over. *KAG* builds a machine-readable
  **knowledge graph**; the *LLM wiki* (Andrej Karpathy) builds human-readable **interlinked
  notes**. Knowledge persists and compounds across queries — paid for with an upfront build
  and the risk it drifts from the source.

The cheap reflex — drop the whole file into ChatGPT/Claude and ask (**direct upload**) —
deserves a fair seat at the table: for a single short PDF it is often the *right* call.
This doc contrasts all six so the choice is deliberate, not reflexive.

## The six options

| | **Direct upload** (whole file in chat) | **Traditional RAG** | **PageIndex** | **Agentic RAG** | **KAG** (knowledge graph) | **LLM wiki** (Karpathy) |
|---|---|---|---|---|---|---|
| Family | retrieve from source | retrieve | retrieve | retrieve | **compile to derived knowledge** | **compile to derived knowledge** |
| How it works | entire PDF → context window; ask | hybrid retrieve top-k chunks → answer | LLM reasons over TOC tree → load sections → answer | LLM tool-loop: search / read / verify | docs → knowledge graph (entities/events) + graph↔text index; query planned as a *logical form*, solved by graph/retrieval/semantic/math operators | agent pre-compiles docs → interlinked markdown pages + index; query navigates the compiled summaries → reads pages |
| Setup & infra | none (a chat UI) | parser + embeddings + vector/lexical index | parser + heading tree (no vectors) | the above + tool loop | heavy — OpenSPG/graph store, Docker, extraction + schema pipeline | light tooling (markdown + Obsidian + a curating agent, e.g. Claude Code) + ongoing curation |
| Upfront build cost | none | low (embed once) | low (extract tree) | low–med (same index) | **high** — graph extraction, schema, alignment | **high** — agent compiles, then maintains |
| LLM calls / query | 1 (huge prompt) | 1 | 1–2 | 3–8 | 3–8 (plan + reasoning operators) | 1–few (index → pages); heavy at *build* time |
| Token cost / query | **high** — re-sends the whole doc every time | low (a few chunks) | low–med (TOC + a few sections) | med–high (loop) | med (graph + matched text); build cost is separate | low (compiled summaries); cost shifts to build/maintenance |
| Latency | secs (grows with doc size) | ~1–2 s | ~3–5 s | 5–30 s | 5–30 s (multi-step solve) | secs (reads compiled pages) |
| Scales to large / many docs | ❌ caps at the context window | ✅ index scales | ✅ tree per doc; descend hierarchically | ✅ (add a `list_documents` tool) | ✅ graph scales across the corpus | ✅ structure stays navigable (~100 pages / 400k words for Karpathy's own); curation, not retrieval, is the cost |
| Citations | weak — model *says* a page; no guarantee | chunk → page + bbox | **section → page + bbox** (most human-legible) | either | graph node/triple → source chunk → page (structured, graph-mediated) | cites the wiki page's noted sources — a layer removed from the raw doc |
| Mechanical quote verification | ❌ none | ✅ code-checked | ✅ code-checked | ✅ + agent self-check | ✅ via mutual graph↔text index back to source | ❌ prose summaries, not verbatim spans |
| Hallucination / abstention control | weak — tends to answer something | medium (constrained prompt) | medium | **best** (can re-search, then abstain) | strong — logical-form + symbolic constraints curb guessing | medium — grounded in curated pages, but summarization can drift |
| Cross-refs & contradictions (e.g. the amendment that supersedes §1) | **good** if it all fits in context | poor (may retrieve only §1, miss the amendment) | good (reasoning can follow the amendment node) | **best** (can search "amendment/superseded") | strong — graph edges link facts, multi-hop; supersession/time needs schema modeling | strong **by design** — the build step links concepts and flags contradictions (when rebuilt) |
| Auditability | none (a chat log) | retrieval scores | reasoning trace | full action trace | logical-form trace + graph path | page edit history + links (git); not a per-query trace |
| Data governance (HR docs) | whole doc egresses to the provider **every query** | only matched chunks egress | TOC + matched sections egress | matched content egresses | matched graph + text egress; **self-hostable** (Docker) | source + wiki stay on disk; only what you send the agent egresses; **self-hostable** |
| Knowledge persistence | none (ephemeral chat) | index only (no synthesis) | tree only | none (per-query) | durable graph (machine-structured) | durable prose; **compounds** over time |
| Eval tractability | hard | **best** (recall@k) | medium (node-set F1) | hardest | hard (graph + reasoning) | hard (open-ended synthesis) |
| Main risk | cost, context limits, silent hallucination | chunking errors | heading-extraction errors | cost / latency / nondeterminism | extraction/schema errors; construction cost & complexity | staleness + summary drift from the source of record |

## The contradiction trap, concretely

The handbook says remote work is "up to **3 days/week** (Tue/Wed/Thu)" in §1, then an
**URGENT AMENDMENT (June 2024)** on the last page supersedes it to "**1 day/week**, must
be a Friday." Asked *"how many days can I work remotely?"*:

- **Direct upload** — usually catches it (both passages are in context) **but** may also
  confidently average them, miss the date, or invent a citation. No mechanical proof.
- **Traditional RAG** — likely retrieves the §1 chunk ("3 days") and **misses** the
  amendment unless the query happens to surface it. The dangerous failure: a confident,
  cited-looking, *wrong* answer.
- **PageIndex** — the amendment is its own TOC node; the reasoning step can pull it in,
  and its summary flags "supersedes Section 1."
- **Agentic RAG** — can issue a second search ("remote work amendment"), read both, and
  reconcile — the most robust, at the most cost.
- **KAG** — the amendment becomes a node/edge; logical-form reasoning can traverse to it —
  *if* the schema models "supersedes"/effective-date. Naive extraction may instead leave
  two conflicting facts in the graph for knowledge alignment to reconcile.
- **LLM wiki** — best case for this trap: ingesting the amendment **updates** the "Remote
  work" page and flags the conflict ("June 2024 amendment supersedes §1: now 1 day/week,
  Fridays"). The catch is **staleness** — if the wiki hasn't been rebuilt since the
  amendment landed, it confidently serves the old "3 days."

This is the case where the cheapest retrieve path is the *most* dangerous, where mechanical
quote verification + abstention earn their keep, and where the compiled approaches help only
if their derived copy is kept current.

## When to use which

- **Direct upload** — one-off question, a single short document, a human reading the
  answer who will sanity-check it. No infra, instant. Don't build a pipeline for this.
- **Traditional RAG** — high query volume, latency/cost-sensitive, mostly single-hop
  questions; you want recall@k you can measure. The recall baseline.
- **PageIndex** — well-structured handbooks/manuals where section-level citations match
  how reviewers think; cheap index, auditable retrieval. **PolicyPal's default.**
- **Agentic RAG** — multi-part or cross-referencing questions, or when an auditable
  action trace per answer is worth the latency. Route to it only when needed.
- **KAG** — large professional/vertical knowledge bases (regulations, medicine, finance)
  whose value is **multi-hop logical reasoning** over interconnected facts, where you can
  invest in schema + graph construction and want self-hosted symbolic+semantic reasoning.
  Overkill for one handbook. ([openspg/kag](https://github.com/openspg/kag))
- **LLM wiki** — a knowledge base a human team curates and reads, that grows over time and
  benefits from **synthesis across many sources**; when a durable, browsable, compounding
  artifact beats per-query freshness (research, personal/team knowledge). Weaker where
  verbatim citation to a live source-of-record and freshness are mandatory.
  ([Karpathy's LLM wiki](https://www.kunalganglani.com/blog/llm-wiki-karpathy-local-knowledge-base))

## The trade you can't dodge

The six fall along one axis. **Retrieve from the source** (Direct upload, Traditional RAG,
PageIndex, Agentic RAG) keeps the original as the source of record — best for verbatim
citation, freshness, and mechanical verification. **Compile into derived knowledge** (KAG,
LLM wiki) buys synthesis, multi-hop reasoning, and an artifact that compounds — at the price
of an upfront build and the risk the derived copy drifts from the source.

For HR policy — where the answer must cite the *current* clause and be mechanically grounded
— a retrieve-from-source path with **mechanical citation verification**, **enforced
abstention**, and **an audit trail** stays PolicyPal's spine: the difference between "sounds
right" and "verifiably grounded." The compiled approaches (KAG, LLM wiki) are how you'd
layer reasoning or a browsable knowledge base *on top* — not how you'd answer "what's the
current remote-work rule."

---

### Sources

- KAG — Knowledge Augmented Generation: <https://github.com/openspg/kag>
- Andrej Karpathy's LLM wiki — explainers:
  [DAIR.AI](https://academy.dair.ai/blog/llm-knowledge-bases-karpathy) ·
  [Data Science Dojo](https://datasciencedojo.com/blog/llm-wiki-tutorial/) ·
  [Level Up Coding](https://levelup.gitconnected.com/beyond-rag-how-andrej-karpathys-llm-wiki-pattern-builds-knowledge-that-actually-compounds-31a08528665e)
