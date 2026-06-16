# PolicyPal: Implementation Guide — Traditional RAG, PageIndex, Agentic RAG

Three retrieval architectures over `Omnicorps_Employee_Handbook.pdf`, sharing one parsing layer and one citation/verification contract. The contract is the trust mechanism; the retrieval strategy is swappable.

## 0. Framework Choices

| Concern | Choice | Rejected alternatives & why |
|---|---|---|
| PDF parsing | **PyMuPDF (fitz)** | Docling/Unstructured: heavier deps, slower; PyMuPDF gives char offsets + bounding boxes needed for highlight rendering |
| Orchestration | **None (plain Python)** | LangChain/LlamaIndex: abstraction tax, opaque prompts, hard to enforce custom citation contract; a single handbook doesn't justify them |
| Vector store | **pgvector** (or in-memory NumPy for prototype) | Pinecone/Chroma: external service / extra dep for <1k chunks is overkill |
| Lexical search | **Postgres `tsvector`** (or `rank_bm25` in-memory) | Elasticsearch: operational overhead unjustified at this scale |
| LLM SDK | **Anthropic / OpenAI SDK directly** | Direct SDK keeps the citation-enforcing prompts inspectable |
| PageIndex | **Self-hosted `VectifyAI/PageIndex` (MIT) for tree generation; own retrieval loop** | Hosted PageIndex API: external data egress of HR documents — likely a compliance non-starter |
| Agent loop | **Hand-rolled tool loop (~80 lines)** | OpenAI Agents SDK / LangGraph: fine, but a verification-critical loop should be fully auditable |
| UI (optional) | Streamlit + PyMuPDF page render with bbox highlights | — |

```
pip install pymupdf anthropic pgvector psycopg[binary] rank_bm25 numpy
```

## 1. Shared Foundation (used by all three)

### 1.1 Structured extraction with provenance

Every downstream citation depends on capturing `{page, section_path, char_span, bbox}` at parse time. This is non-negotiable: you cannot retrofit bounding boxes after chunking.

```python
# parse.py
import fitz, re
from dataclasses import dataclass, field

@dataclass
class Block:
    text: str
    page: int            # 1-based
    bbox: tuple          # (x0, y0, x1, y1) on page — enables highlight rendering
    font_size: float
    is_heading: bool

def extract_blocks(pdf_path: str) -> list[Block]:
    doc = fitz.open(pdf_path)
    body_size = _modal_font_size(doc)          # most common size = body text
    blocks = []
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:                  # skip images
                continue
            spans = [s for l in b["lines"] for s in l["spans"]]
            if not spans:
                continue
            text = " ".join(s["text"] for s in spans).strip()
            size = max(s["size"] for s in spans)
            bold = any(s["flags"] & 16 for s in spans)
            blocks.append(Block(
                text=text, page=pno, bbox=b["bbox"], font_size=size,
                is_heading=(size > body_size * 1.15) or (bold and len(text) < 80),
            ))
    return blocks

def _modal_font_size(doc) -> float:
    from collections import Counter
    sizes = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0: continue
            for l in b["lines"]:
                for s in l["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"])
    return sizes.most_common(1)[0][0]
```

**Design note — heading detection is the fragile step.** Font-size heuristics fail on handbooks that use color or numbering instead of size. Validate by printing the inferred outline and eyeballing it against the PDF's TOC before proceeding. If `doc.get_toc()` returns a non-empty embedded TOC, prefer it outright — it is ground truth.

### 1.2 The citation contract (shared output schema)

All three pipelines must emit the same structure, so the verification layer and UI are written once:

```python
# contract.py
from dataclasses import dataclass

@dataclass
class Citation:
    chunk_id: str        # or tree node_id for PageIndex
    page: int
    section_path: str    # e.g. "4 Leave Policies > 4.2 Bereavement Leave"
    quote: str           # verbatim span the model claims supports the answer

@dataclass
class Claim:
    text: str            # one atomic statement in the answer
    citations: list[Citation]
    verified: bool = False   # set by the verifier, never by the LLM

@dataclass
class Answer:
    claims: list[Claim]
    abstained: bool      # True => "not found in handbook"
```

### 1.3 Mechanical quote verification (shared)

The single highest-leverage trust feature. The LLM asserts a quote; code — not another LLM — checks it exists in the cited source. Runs identically over chunks (traditional/agentic) or tree-node text (PageIndex).

```python
# verify.py
import re
from difflib import SequenceMatcher

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def verify_quote(quote: str, source_text: str, fuzz: float = 0.92) -> bool:
    q, src = normalize(quote), normalize(source_text)
    if q in src:
        return True
    # fuzzy fallback: tolerate hyphenation/ligature artifacts from PDF extraction
    m = SequenceMatcher(None, q, src).find_longest_match(0, len(q), 0, len(src))
    return m.size / max(len(q), 1) >= fuzz

def verify_answer(answer, get_source_text) -> "Answer":
    for claim in answer.claims:
        claim.verified = bool(claim.citations) and all(
            verify_quote(c.quote, get_source_text(c)) for c in claim.citations
        )
    return answer
```

A claim with `verified=False` is rendered with a warning flag, never silently shown. Numeric claims warrant a stricter rule: extract digits from the claim and require them to appear in the cited quote.

---

## 2. Approach 1 — Traditional RAG

### 2.1 Design

```
PDF → blocks → section-aware chunks (+metadata) → embed → pgvector
                                    └→ tsvector (lexical)
query → [dense top-k] ⊕ [BM25 top-k] → RRF fuse → top-n → generate(JSON claims) → verify → render
```

Rationale for hybrid: HR queries are terminology-dense ("bereavement", "FMLA", "401k vesting") — exact lexical match is often the strongest signal; dense retrieval covers paraphrase ("my dad died, what am I entitled to?"). Reciprocal Rank Fusion needs no score calibration between the two.

### 2.2 Chunking

```python
# chunk.py
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int
    section_path: str
    bboxes: list          # [(page, bbox), ...] for every block in the chunk

def chunk_by_section(blocks, max_tokens=450, overlap_blocks=1):
    chunks, path, buf = [], [], []
    def flush():
        if not buf: return
        text = "\n".join(b.text for b in buf)
        chunks.append(Chunk(
            id=f"c{len(chunks):04d}",
            text=f"[{' > '.join(path)}]\n{text}",   # prepend section path: helps both retrieval & the LLM
            page_start=buf[0].page, page_end=buf[-1].page,
            section_path=" > ".join(path),
            bboxes=[(b.page, b.bbox) for b in buf],
        ))
    for b in blocks:
        if b.is_heading:
            flush(); buf = []
            path = _update_path(path, b)           # maintain heading hierarchy by font size
            continue
        buf.append(b)
        if _tok(buf) > max_tokens:
            flush(); buf = buf[-overlap_blocks:]
    flush()
    return chunks
```

Heading-bounded chunks mean a chunk never straddles two policies — the dominant failure mode of fixed-size chunking on policy documents (a chunk containing the tail of "Vacation" and the head of "Sick Leave" produces confidently wrong day counts).

### 2.3 Indexing & hybrid retrieval (pgvector + tsvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
    id text PRIMARY KEY,
    text text NOT NULL,
    section_path text,
    page_start int, page_end int,
    bboxes jsonb,
    embedding vector(1024),                       -- MUST match embedding model dim
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX ON chunks USING gin(tsv);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

```python
# retrieve.py
def hybrid_search(conn, query: str, qvec, k_each=20, k_final=6):
    dense = conn.execute(
        "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (qvec, k_each)).fetchall()
    lex = conn.execute(
        "SELECT id FROM chunks WHERE tsv @@ plainto_tsquery('english', %s) "
        "ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC LIMIT %s",
        (query, query, k_each)).fetchall()
    return rrf([r[0] for r in dense], [r[0] for r in lex])[:k_final]

def rrf(*rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

### 2.4 Citation-constrained generation

```python
GEN_SYSTEM = """You answer HR policy questions using ONLY the provided handbook excerpts.
Return JSON only:
{"abstained": bool,
 "claims": [{"text": "...", "citations": [{"chunk_id": "...", "quote": "<VERBATIM text copied from that chunk>"}]}]}
Rules:
- Every claim must carry >=1 citation. quote must be copied character-for-character from the chunk.
- If the excerpts do not contain the answer, set abstained=true and claims=[].
- Never use knowledge outside the excerpts."""

def generate(client, query, chunks):
    ctx = "\n\n".join(f"<chunk id='{c.id}' section='{c.section_path}' pages='{c.page_start}-{c.page_end}'>\n{c.text}\n</chunk>" for c in chunks)
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1500,
        system=GEN_SYSTEM,
        messages=[{"role": "user", "content": f"{ctx}\n\nQuestion: {query}"}],
    )
    return parse_answer(resp.content[0].text)      # json.loads with fence stripping
```

### 2.5 Properties

| Strength | Weakness |
|---|---|
| One LLM call per query → cheapest, lowest latency (~1–2 s) | Single-shot retrieval: multi-part questions ("bereavement leave AND counseling services") may retrieve only one facet |
| Stateless, simple to evaluate (recall@k on chunk IDs) | Misses cross-references ("see Section 7.3") — no similarity between pointer and target |
| Hybrid handles both jargon and paraphrase | Embedding dim locked at table creation; chunking quality silently bounds everything downstream |

---

## 3. Approach 2 — PageIndex (Vectorless, Reasoning-Based)

### 3.1 Design

PageIndex (VectifyAI, MIT-licensed) replaces similarity search with LLM reasoning over a hierarchical table-of-contents tree. No embeddings, no chunking, no vector DB. Retrieval = tree search: the LLM reads node titles/summaries and decides which sections plausibly contain the answer, the way a human navigates a handbook's TOC.

```
PDF → PageIndex tree build (one-time, LLM-assisted) → tree JSON
query → LLM tree-search over node titles/summaries → selected node_ids
      → load full node text → generate(JSON claims) → verify → render
```

Why it fits this problem unusually well:

| Property | PolicyPal relevance |
|---|---|
| Retrieval unit = real document section | Citations are inherently human-meaningful: "Section 4.2 Bereavement Leave, p. 23" — exactly what an HR specialist would write themselves |
| Retrieval is a traceable reasoning path | The tree path itself ("Leave Policies → Bereavement") is shown to the reviewer; auditable, unlike opaque cosine scores |
| No chunk-boundary errors | A policy section is never split mid-table |
| Handles cross-references | The reasoning step can follow "see Appendix B" by navigating to Appendix B — flat similarity search cannot |
| No embedding infra | No dimension-lock problem, no re-embedding on model change |

Cost profile inverts vs. traditional RAG: cheap index, expensive queries (1–3 LLM calls per retrieval).

### 3.2 Tree construction

Option A — use the repo (handles TOC detection, large-doc splitting, node summarization):

```bash
git clone https://github.com/VectifyAI/PageIndex && cd PageIndex
pip install -r requirements.txt
python run_pageindex.py --pdf_path Omnicorps_Employee_Handbook.pdf
# → results/Omnicorps_Employee_Handbook_structure.json
```

Option B — minimal self-built tree (sufficient for one handbook; reuses Section 1.1 blocks). Worth doing to keep the stack dependency-free and fully understood:

```python
# tree_build.py
import json

def build_tree(blocks, client):
    """Heading hierarchy -> tree; LLM writes a 1-2 sentence summary per node."""
    root = {"node_id": "root", "title": "Employee Handbook", "children": [], "page_start": 1}
    stack = [(0.0, root)]   # (font_size, node)
    for b in blocks:
        if b.is_heading:
            node = {"node_id": f"n{id(b)%10**6}", "title": b.text,
                    "page_start": b.page, "text_blocks": [], "children": []}
            while len(stack) > 1 and stack[-1][0] <= b.font_size:
                stack.pop()
            stack[-1][1]["children"].append(node)
            stack.append((b.font_size, node))
        else:
            stack[-1][1].setdefault("text_blocks", []).append(b)
    _summarize(root, client)   # one batched LLM call: node texts -> {node_id: summary}
    return root
```

Each node retains its `text_blocks` (with pages + bboxes), so the citation contract and highlight rendering work unchanged.

### 3.3 Reasoning-based retrieval

```python
# tree_search.py
SEARCH_PROMPT = """You are searching an employee handbook for sections that answer a question.
Below is the handbook's tree structure (node_id, title, summary, pages).
Return JSON only: {"thinking": "...", "node_ids": ["...", ...]}
Select ALL nodes needed — a question may span several policies (e.g. leave days AND support services).
If a selected section references another section, include the referenced node too.
If no node can answer, return {"node_ids": []}."""

def tree_search(client, query, tree) -> list[str]:
    toc = render_toc(tree)   # indented "node_id | title | summary | pp. x-y" lines
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=800,
        system=SEARCH_PROMPT,
        messages=[{"role": "user", "content": f"<toc>\n{toc}\n</toc>\n\nQuestion: {query}"}],
    )
    out = parse_json(resp.content[0].text)
    log_retrieval_trace(query, out)        # persist "thinking" — this IS the audit trail
    return out["node_ids"]
```

An employee handbook (~50–150 pages) yields a TOC small enough to fit in one prompt — single-call tree search, no iterative descent needed. For larger corpora, descend level-by-level (ask which top-level chapters are relevant, then recurse into their children).

Generation then reuses `generate()` from §2.4 with node texts in place of chunks; `verify_answer()` checks quotes against node text.

### 3.4 Properties

| Strength | Weakness |
|---|---|
| Citations align with document structure → fastest human verification | Wholly dependent on heading-extraction quality; a flat/badly-structured PDF degrades it to nothing |
| Retrieval reasoning is logged and reviewable | Higher per-query cost/latency than vector lookup |
| Robust to paraphrase via genuine reasoning, not geometry | Summaries can mislead the search LLM (a summarization error is a retrieval error) |
| Follows in-document references | Retrieval recall is not benchmarkable with classic @k metrics in the same way; evaluate as node-set F1 |

---

## 4. Approach 3 — Agentic RAG

### 4.1 Design

The previous two are fixed pipelines. Agentic RAG hands the LLM **tools** and lets it run a retrieve–assess–retrieve loop until it can answer or concludes it cannot. The agent composes the other two approaches' primitives.

```
query → agent loop:
          ├─ decompose question (implicit, in reasoning)
          ├─ search_handbook(q)          # hybrid search from §2 — broad recall
          ├─ browse_toc() / read_section(node_id)   # tree from §3 — structure-aware
          ├─ verify_quote(quote, id)     # the agent self-checks BEFORE answering
          └─ submit_answer(claims) | abstain(reason)
        → server-side verify (§1.3 — never trust the agent's own check) → render
```

Design decisions:
- **Tools mirror what an HR specialist actually does**: keyword-search the PDF, scan the TOC, read a section, double-check a quote. This makes agent traces legible to the HR reviewer — each step is an action a human recognizes.
- **`verify_quote` exposed as a tool AND re-run server-side.** Letting the agent check its own quotes reduces retries; re-verifying outside the loop closes the self-grading hole.
- **Hard iteration cap (6–8 tool calls)** and forced `submit_answer | abstain` terminal action — bounds cost and prevents loops.

### 4.2 Tool definitions and loop

```python
# agent.py
TOOLS = [
  {"name": "search_handbook",
   "description": "Hybrid keyword+semantic search. Returns top chunks with ids, section paths, pages.",
   "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
  {"name": "browse_toc",
   "description": "Return the handbook's table of contents with node ids and summaries.",
   "input_schema": {"type": "object", "properties": {}}},
  {"name": "read_section",
   "description": "Return full text of a TOC node.",
   "input_schema": {"type": "object", "properties": {"node_id": {"type": "string"}},
                    "required": ["node_id"]}},
  {"name": "verify_quote",
   "description": "Check a verbatim quote exists in chunk/node `source_id`. Use before submitting.",
   "input_schema": {"type": "object", "properties": {
       "quote": {"type": "string"}, "source_id": {"type": "string"}},
       "required": ["quote", "source_id"]}},
  {"name": "submit_answer",
   "description": "Final answer as claims with citations (chunk/node id + verbatim quote).",
   "input_schema": CLAIMS_SCHEMA},
  {"name": "abstain",
   "description": "Use when the handbook does not contain the answer.",
   "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}}},
]

AGENT_SYSTEM = """You answer Omnicorps HR policy questions strictly from the Employee Handbook, via tools.
Method: break the question into the distinct policies it touches; search or browse the TOC for each;
read full sections rather than relying on snippets; follow cross-references; verify every quote with
verify_quote before submit_answer. If evidence is missing or contradictory, abstain and say why.
You have at most 8 tool calls."""

def run_agent(client, query, exec_tool, max_steps=8):
    msgs = [{"role": "user", "content": query}]
    trace = []
    for _ in range(max_steps):
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000,
                                      system=AGENT_SYSTEM, tools=TOOLS, messages=msgs)
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            break
        msgs.append({"role": "assistant", "content": resp.content})
        results = []
        for c in calls:
            trace.append({"tool": c.name, "input": c.input})
            if c.name in ("submit_answer", "abstain"):
                return finalize(c, trace)          # -> Answer + audit trace
            results.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": exec_tool(c.name, c.input)})
        msgs.append({"role": "user", "content": results})
    return abstained_answer("step budget exhausted", trace)
```

### 4.3 Properties

| Strength | Weakness |
|---|---|
| Handles multi-facet questions natively ("services AND vacation for bereaved employee" → two searches) | Most expensive and slowest (3–8 LLM calls); latency seconds-to-tens-of-seconds |
| Self-correcting: empty search → reformulate; failed quote check → re-read source | Non-deterministic paths complicate evaluation and caching |
| Full action trace = strongest audit story | More failure surface (loops, tool misuse); needs caps and timeouts |
| Gracefully extends to multi-document future (add a `list_documents` tool) | Over-kill for single-hop questions, which are the majority |

---

## 5. Comparison & Recommendation

| Dimension | Traditional RAG | PageIndex | Agentic RAG |
|---|---|---|---|
| LLM calls / query | 1 | 1–2 (+index build) | 3–8 |
| Latency | ~1–2 s | ~3–5 s | 5–30 s |
| Infra | Postgres+pgvector, embedding model | none beyond LLM | both of the left |
| Citation granularity | chunk → page+bbox | section → page+bbox (most human-legible) | either |
| Multi-hop / cross-reference | poor | good | best |
| Auditability of retrieval | scores only | reasoning trace | full action trace |
| Eval tractability | best (recall@k) | medium (node-set F1) | hardest |
| Main risk | chunking errors | heading-extraction errors | cost/latency/nondeterminism |

**Recommendation for PolicyPal:** route by complexity rather than betting on one.

1. **Build the shared layer first** (parse → contract → verify → highlight render). It delivers the trust gain regardless of retrieval strategy and is reusable across all three.
2. **Default path: PageIndex-style retrieval.** A handbook is exactly the structured, TOC-rich document the approach was designed for, and section-level citations match how HR already reasons about the document. Keep §3.2 Option B (self-built tree) to avoid an extra dependency.
3. **Fallback path: hybrid traditional RAG** when tree search returns no nodes, and as the recall baseline in evaluation.
4. **Agentic loop only for detected multi-part questions** (cheap classifier or "contains multiple question facets" prompt check) — or as a v2 once single-hop trust is established.

Evaluate each path separately on the same hand-curated set (~25 Q/A/evidence triples + ~5 unanswerable adversarials): retrieval recall (chunk-ID or node-ID level), citation validity rate (mechanical), faithfulness (LLM-judge, acknowledging position/verbosity/self-preference biases), abstention accuracy.

## 6. Build Order

| Step | Deliverable | Gate before next step |
|---|---|---|
| 1 | `parse.py` — blocks with page/bbox/headings | Inferred outline matches PDF TOC on visual inspection |
| 2 | `contract.py`, `verify.py` | Unit tests incl. hyphenation/whitespace edge cases |
| 3 | Traditional RAG end-to-end | ≥0.9 recall@6 on curated set |
| 4 | Tree build + PageIndex retrieval | Node-set F1 ≥ traditional recall on same set |
| 5 | Router + agentic loop | Multi-part questions answered with all facets cited |
| 6 | Streamlit UI: claim ↔ highlighted PDF region | HR specialist can verify an answer in <15 s |
| 7 | README incl. AI-tool disclosure section | — |
