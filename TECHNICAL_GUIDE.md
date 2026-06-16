# PolicyPal: Comprehensive Technical Guide

**Version:** 1.0  
**Last Updated:** June 2024  
**Author:** PolicyPal Development Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Three RAG Approaches](#three-rag-approaches)
4. [Retrieval & Embedding Parameters](#retrieval--embedding-parameters)
5. [Framework & Technology Stack](#framework--technology-stack)
6. [Core Modules](#core-modules)
7. [Docker Deployment](#docker-deployment)
8. [Installation & Setup](#installation--setup)
9. [Running the System](#running-the-system)
10. [Evaluation & Results](#evaluation--results)
11. [Performance Breakdown](#performance-breakdown)
12. [Future Improvements](#future-improvements)
13. [Troubleshooting](#troubleshooting)

---

## Executive Summary

**PolicyPal** is a production-ready Retrieval-Augmented Generation (RAG) system designed to answer questions about employee handbooks with verifiable citations. It implements three distinct retrieval architectures over a shared parsing and citation-verification layer:

1. **Traditional RAG** — Fast hybrid search (dense + keyword) with single-pass generation
2. **PageIndex RAG** — Tree-based reasoning over document structure (no embeddings)
3. **Agentic RAG** — Multi-step tool loop with self-correction (most sophisticated)

The system prioritizes **citation correctness** through mechanical (not LLM-based) quote verification, ensuring every claim can be traced to its source with bounding-box accuracy for rendering highlights in the PDF.

### Key Features

- **3 retrieval strategies** with unified output schema
- **Mechanical verification** — code validates every quote, not an LLM
- **Section-aware chunking** — chunks never straddle policy boundaries
- **Hybrid search** — dense embeddings + BM25 keyword search (Reciprocal Rank Fusion)
- **Postgres + pgvector** — production-grade vector store with full-text search
- **Cost tracking** — per-query token counts and estimated costs
- **Offline testing** — mock provider runs the entire pipeline with no API keys
- **Multi-provider support** — Anthropic, OpenAI, OpenRouter APIs

---

## Architecture Overview

### Shared Foundation (All Three Approaches)

Every retrieval strategy builds on this immutable spine:

```
PDF (Handbook)
    ↓
parse.py (PyMuPDF)
    ↓ [Block: page, bbox, heading, font]
    ├─→ chunk.py (section-aware chunking)
    │   ↓ [Chunk: id, text, section_path, page_start, page_end, bboxes]
    │
    ├─→ tree.py (outline + node summaries)
    │   ↓ [TreeNode: id, title, summary, section_path, page_start/end, children]
    │
    └─→ store.py (dense + lexical indexing)
        ↓ [InMemoryStore or PgVectorStore]

contract.py (Shared Output Schema)
    ├─ Answer: claims + abstained + retrieval + usage + cost_usd + latency_s
    ├─ Claim: text + citations + verified flag
    └─ Citation: source_id, page, section_path, quote, page_end

verify.py (Mechanical Quote Verification)
    ↓ normalize & fuzzy-match quotes against source
```

**Design philosophy:** Capture `{page, section_path, bbox}` at parse time. These immutable provenance values flow through all three approaches, so citations and highlight rendering are written once.

---

### Query Routing

```
Question
    ↓
approach toggle?
    ├─ explicit ("traditional"|"pageindex"|"agentic") → run that one, done
    └─ "auto" (default)
            ↓
        detect multi-facet? (and / or / ; / also / both)   ← regex heuristic
            ├─ YES → agentic (handles complex decomposition)
            └─ NO → pageindex (default, good for single facets)
                    ├─ success → return Answer
                    └─ abstain → traditional (recall fallback)
```

The router is a **keyword heuristic, not an ML classifier** — one regex over the
lowercased question. `\b...\b` word boundaries prevent false hits (e.g. "st**and**ard"
does not match `\band\b`). The actual implementation
([`policypal/pipeline.py`](policypal/pipeline.py)):

```python
def route(self, query: str) -> str:
    multi = bool(re.search(r"\band\b|\bor\b|;|\balso\b|both", query.lower()))
    return "agentic" if multi else "pageindex"

def answer_routed(self, query: str, approach: str = "auto") -> Answer:
    """approach='auto' → router picks (with PageIndex→traditional fallback);
    any explicit approach bypasses the router. The approach actually used is
    recorded in ans.retrieval['approach']."""
    if approach not in ("auto", "traditional", "pageindex", "agentic"):
        raise ValueError(...)
    if approach != "auto":                      # toggle: user forced one
        ans = self.answer(query, approach)
        ans.retrieval["approach"] = approach
        return ans
    chosen = self.route(query)
    ans = self.answer(query, chosen)
    if chosen == "pageindex" and ans.abstained:  # recall fallback
        ans = self.answer(query, "traditional")
    ans.retrieval["approach"] = ...
    return ans
```

> **Note the asymmetry:** the abstain→traditional fallback fires *only* on the
> PageIndex path. An `agentic` answer (even an abstention) is final. The router's
> blind spots (e.g. "alcohol **and** drugs" → agentic unnecessarily) are why a
> classifier-based upgrade is improvement #14 in [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

---

## Three RAG Approaches

### 1. Traditional RAG — Hybrid Search + Single-Pass Generation

**Speed:** ~1–2 seconds · **Cost:** Cheapest · **Calls:** 1 LLM call

#### Workflow

```
Query
  ├─→ Dense retrieval (embeddings similarity) → top k_each=20
  ├─→ Lexical retrieval (BM25)                → top k_each=20
  ├─→ Reciprocal Rank Fusion (rank fusion)
  ├─→ Top k_final=6 chunks                     ← exactly 6 reach the LLM
  └─→ generate() [claims + quotes]
      └─→ verify_quote() [mechanical check]
          └─→ Answer {✅ verified | ⚠️ flagged}
```

**Chunks per query: exactly 6.** Each retriever proposes 20 candidates
(`k_each=20`); after RRF fusion the top 6 (`k_final=6`) are sent to generation. See
[Retrieval & Embedding Parameters](#retrieval--embedding-parameters) for all values.

#### Strengths

- **Fastest** — single LLM call, ideal for simple factual questions
- **Cost-effective** — minimal token spend
- **Paraphrase coverage** — dense + keyword hybrid catches both lexical matches and semantic paraphrases

#### Weaknesses

- **Single-shot retrieval** — may miss facets of multi-part questions
- **No cross-references** — cannot follow "see Appendix B" style links
- **Query reformulation not automatic** — if retrieval misses the mark, no second chance

#### When to Use

- Single-facet questions ("How many days of bereavement leave?")
- Time-sensitive scenarios where speed is critical
- Cost-constrained deployments

#### Implementation

```python
from policypal import PolicyPal

pal = PolicyPal("handbook.pdf")
ans = pal.answer("How many days of bereavement leave?", 
                approach="traditional")
print(ans.render())
```

---

### 2. PageIndex RAG — Tree-Based Reasoning

**Speed:** ~2–3 seconds · **Cost:** Moderate · **Calls:** 2 LLM calls · **Embeddings:** None

#### Workflow

```
Query
  ├─→ tree_search() [LLM reads TOC + summaries, reasons which sections apply]
  ├─→ node_ids selected (variable count, with "thinking" audit trail)
  ├─→ Load full section text (each node = its whole section + descendants)
  └─→ generate() [claims + quotes]
      └─→ verify_quote()
          └─→ Answer {section reference: "§4.2, p.23"}
```

**Chunks per query: variable, no `k`.** PageIndex does not retrieve fixed-size
chunks — the tree-search LLM selects however many section `node_ids` it judges
relevant (prompt: *"Select ALL nodes needed"*). Each selected node carries its
entire section (own body **plus** all descendant subsections), so even one node can
be a large block of text. Zero nodes selected → the answer abstains.

#### Strengths

- **No embeddings required** — vectorless, fully offline
- **Explicit reasoning** — "thinking" path is logged for audit
- **Section-level citations** — matches how HR already reads the handbook ("Section 4.2, page 23")
- **Follows cross-references** — agent can reason about "see Appendix B"
- **Robust to paraphrase** — doesn't depend on embedding quality

#### Weaknesses

- **Requires good headings** — font-size heuristics fail if PDF uses color/numbering
- **Flat PDFs degrade fast** — no table of contents = no reasoning scaffolding
- **Slow on massive outlines** — large TOCs require many LLM reads

#### When to Use

- Well-structured handbooks with clear outline
- Offline-first deployments (no embedding service needed)
- Compliance scenarios (reasoning path is auditable)

#### Implementation

```python
pal = PolicyPal("handbook.pdf")
ans = pal.answer("Can I carry over unused vacation?", 
                approach="pageindex")
print(f"Section: {ans.claims[0].citations[0].section_path}")
```

---

### 3. Agentic RAG — Bounded Tool Loop

**Speed:** ~5–15 seconds · **Cost:** Highest · **Calls:** 1–8 LLM calls

#### Workflow

```
Query
  ↓
Agent (≤8 tool calls — max_steps=8)
  ├─→ search_handbook() → hybrid search → 6 chunks per call (k_each=20, k_final=6)
  ├─→ browse_toc() → tree navigation
  ├─→ read_section() → load full section
  ├─→ verify_quote() → self-check quotes
  ├─→ [loop until confident or tool limit hit]
  └─→ submit_answer() or abstain
      ↓
      Server-side verify_quote() [never trust agent's own check]
      ↓
      Answer + full action trace
```

**Chunks per query: variable, bounded by the step cap.** Each `search_handbook`
call returns 6 chunks (same `k_each=20 → k_final=6` funnel as Traditional), but the
agent may search/browse/read repeatedly. The ceiling is the **8 tool-call cap**
(`max_steps=8`), so total context seen is variable but bounded.

#### Strengths

- **Self-corrects** — empty search → reformulate; failed quote → re-read
- **Decomposes complex questions** — "I have a kid, planning to leave" → multiple tool calls
- **Most flexible** — can mix search + browse + read + verify
- **Audit trail** — full action history visible
- **Handles contradictions** — can surface both old rule and amendment

#### Weaknesses

- **Most expensive** — 3–8 calls × cost per call
- **Non-deterministic** — different runs may use different paths
- **Tool loop overhead** — slower (seconds to tens of seconds)
- **Requires stricter controls** — hard cap on tool calls prevents infinite loops

#### When to Use

- Complex multi-facet questions
- High-stakes HR decisions (needs full reasoning visible)
- When coverage is more important than cost

#### Implementation

```python
pal = PolicyPal("handbook.pdf")
ans = pal.answer(
    "I have a newborn and may need to take leave. What are my options?",
    approach="agentic"
)
print(f"Tool calls: {len(ans.retrieval['trace'])}")   # the action trace lives in ans.retrieval
```

---

### Approach Toggle & Automatic Routing

`answer_routed(query, approach="auto")` is the single smart entry point. The
`approach` argument is a **toggle**: `"auto"` lets the router decide; any explicit
approach forces that one and skips the router.

```python
# auto (default): router picks + PageIndex→traditional fallback
ans = pal.answer_routed(question)

# forced: bypass the router, run exactly this approach
ans = pal.answer_routed(question, approach="agentic")

# inspect which approach actually ran (useful after an auto fallback)
print(ans.retrieval["approach"])     # e.g. "traditional (fallback from pageindex)"
```

---

## Retrieval & Embedding Parameters

This section consolidates every tunable that affects retrieval and embedding. Values
are grouped by whether they are **environment-configurable** (set in `.env`) or
**hardcoded function defaults** (currently require a code edit).

### How many chunks reach the LLM, by approach

| Approach | Chunks/nodes sent to LLM | Mechanism |
|---|---|---|
| **Traditional** | **Exactly 6** | `hybrid_search(k_each=20, k_final=6)` — [traditional.py](policypal/traditional.py) |
| **PageIndex** | **Variable** (LLM-selected) | tree-search returns N section node_ids; no cap |
| **Agentic** | **6 per `search_handbook` call**, repeatable | `k_each=20, k_final=6`; up to 8 tool calls |

The two-stage funnel (Traditional & Agentic): each of dense + BM25 proposes
`k_each=20` candidates → RRF fuses them → top `k_final=6` survive to generation.

### Retrieval depth — *hardcoded (not env vars yet)*

| Parameter | Value | Location |
|---|---|---|
| `k_each` (candidates per retriever) | `20` | [retrieve.py](policypal/retrieve.py), [traditional.py](policypal/traditional.py), [agent.py](policypal/agent.py) |
| `k_final` (fused chunks → LLM) | `6` | same |
| RRF constant `k` | `60` | [retrieve.py](policypal/retrieve.py) `rrf()` |
| Agent step cap `max_steps` | `8` | [agent.py](policypal/agent.py) `run_agent()` |

### Hybrid search — env-configurable

| Parameter | Env var | **Default** | Notes |
|---|---|---|---|
| BM25 term-frequency saturation | `BM25_K1` | **`1.5`** | higher = term repeats matter more |
| BM25 length normalization | `BM25_B` | `0.75` | 0 = none, 1 = full |
| Keyword stemming | `KEYWORD_STEMMING` | `true` | Snowball stemmer in the analyzer |
| Fusion method | `HYBRID_METHOD` | `rrf` | `rrf` \| `weighted` |
| Dense weight (weighted only) | `HYBRID_ALPHA` | `0.5` | fraction given to dense vs lexical |

### Chunking — *hardcoded build-time defaults*

| Parameter | Value | Notes |
|---|---|---|
| `max_tokens` per chunk | `450` | approx, ~4 chars/token ≈ 1,800 chars |
| `overlap_blocks` | `1` | **intra-section only** — never overlaps across a heading boundary |
| section boundary | heading-bounded | a chunk never straddles two policies |

What gets embedded/indexed is the chunk's full text **including a prepended
`[section_path]` header** ([chunk.py](policypal/chunk.py)), which anchors both dense
and keyword retrieval on the section.

### Embeddings

> **Only Traditional and Agentic use embeddings** (the dense half of hybrid search).
> **PageIndex is vectorless** — it reasons over the TOC and embeds nothing.

| Parameter | Env var | Default | Location |
|---|---|---|---|
| Provider | `EMBEDDING_PROVIDER` | `local` (offline hashing) | [config.py](policypal/config.py) |
| Model | `EMBEDDING_MODEL` | `text-embedding-3-small` | [config.py](policypal/config.py) |
| Dimension (request/fallback) | `EMBEDDING_DIM` | `1536` | probed live for API models |
| API batch size | — (hardcoded) | `100` per request | [embeddings.py](policypal/embeddings.py) |

**The three embedders** ([embeddings.py](policypal/embeddings.py)):

| Backend | Default model | Dimension | Behavior |
|---|---|---|---|
| `local` (default) | `HashingEmbedder` | **1536** | feature-hashing bag-of-words, signed buckets, sublinear tf; deterministic, offline; strong on jargon, weak on pure paraphrase |
| `openai` | `text-embedding-3-small` | **1536** (probed) | real API embeddings, batched 100/req |
| `openrouter` | `openai/text-embedding-3-small` | **1536** (probed) | same class via OpenRouter gateway |


- **One vector per chunk** — no multi-vector/sliding-window scheme. The in-memory
  store is a single `(N, D)` float32 matrix (N = chunk count).
- **All vectors are L2-normalized**, so a dot product equals cosine similarity. Dense
  search is `matrix @ q` (in-memory) or the `<=>` cosine operator (pgvector).
- **Matryoshka shortening:** `text-embedding-3-*` and `qwen3-embedding` honor a
  requested `dimensions`; set `EMBEDDING_DIM` to ask for a shorter vector. Otherwise
  the real dim is **probed once** from the live model (the 1536 in config is only a
  fallback if the probe fails).

### Overlap — summary (a common point of confusion)

| Kind of overlap | Present? | Detail |
|---|---|---|
| **Chunk text overlap** | Yes — 1 block | only when an oversized section is split mid-section; **never across a heading boundary** |
| **Vector overlap** | No | exactly one embedding per chunk |

### Vector storage & indexing

| | In-memory ([store.py](policypal/store.py)) | pgvector ([pgstore.py](policypal/pgstore.py)) |
|---|---|---|
| Vector storage | `(N, D)` float32 NumPy matrix | `vector(D)` column |
| Dense index | none (brute-force `matrix @ q`) | **HNSW**, `vector_cosine_ops` — *only if D ≤ 2000* |
| If D > 2000 | n/a | stores vectors, skips ANN index → exact sequential cosine scan |
| Keyword half | BM25 (`rank_bm25`) | Postgres FTS (`tsvector` + GIN, `ts_rank_cd`) |
| Similarity metric | cosine | cosine (`1 - (embedding <=> q)`) |

**Caveats:** pgvector's HNSW caps at **2000 dims** — `text-embedding-3-large` (3072)
falls back to an exact scan (instant at handbook scale; a concern at corpus scale —
see [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) #7). No custom HNSW
`m`/`ef_construction`/`ef_search` are set (pgvector defaults). pgstore drops & rebuilds
the table if the embedder's dimension changes, since a `vector(D)` column is
fixed-width.

### LLM sampling parameters — env-configurable

| Parameter | Env var | Default | Notes |
|---|---|---|---|
| Temperature | `TEMPERATURE` | `0` (default) | |
| top_p | `TOP_P` | `1` | |
| seed | `SEED` | `114514` | |
| `max_tokens` | `MAX_TOKENS` | `2000` | tree-search overrides to `800`; agent loop uses `2000`/call |

---

## Framework & Technology Stack

### Language & Runtime

- **Python 3.12** — modern async/type-hint support
- **No framework required** — plain Python for full auditability

### Core Dependencies

| Layer | Package | Version | Purpose |
|-------|---------|---------|---------|
| **PDF Parsing** | `pymupdf` (fitz) | Latest | Extract text + bboxes for highlight rendering |
| **LLM Providers** | `anthropic` | Latest | Anthropic Claude API |
| | `openai` | Latest | OpenAI GPT API |
| | `openrouter` | *via openai-compatible* | OpenRouter gateway |
| **Vector Store** | `pgvector` | Latest | Postgres vector extension |
| | `psycopg[binary]` | Latest | Postgres driver |
| **Lexical Search** | `rank_bm25` | Latest | BM25 keyword search (no Elasticsearch) |
| **Numerics** | `numpy` | Latest | Dense vector operations (in-memory) |
| **Testing** | `pytest` | Latest | Test framework |
| **Notebook** | `jupyterlab` | Latest | Interactive environment |
| **UI** (planned) | `streamlit` | Latest | Installed for a *planned* PDF highlight viewer — see Improvements #12; no `app.py` exists yet |

**Rationale for each choice:**

- **PyMuPDF vs. Unstructured/Docling:** PyMuPDF is lightweight, gives bounding boxes (needed for highlights), and is ~2x faster. Unstructured/Docling pull heavy dependencies for scope we don't need.
  
- **No LangChain/LlamaIndex:** These frameworks hide the prompt, making citation verification hard to audit. At handbook scale (<5k chunks), hand-rolled orchestration is shorter and more transparent.

- **pgvector vs. Pinecone/Weaviate:** For <2k chunks, a local database is simpler, cheaper, and compliant (no data egress). Pinecone/Weaviate add operational overhead unjustified at this scale.

- **BM25 (rank_bm25) vs. Elasticsearch:** No need for a separate service. `rank_bm25` is in-RAM and sufficient; Postgres `tsvector` for production (both implemented in `pgstore.py`).

---

## Core Modules

### `parse.py` — PDF Parsing with Provenance

**Responsibility:** Extract structured blocks from PDF with page number, bounding box, and heading detection.


**Design Notes:**
- Heading detection is the fragile step (§ Improvements #4)
- Currently uses **font-size heuristics only**: a block is a heading if its size >
  1.15× the modal (body) size, *or* it's bold and under 80 chars. These fail on PDFs
  that signal headings by colour/numbering.
- Preferring `doc.get_toc()` when an embedded TOC exists is a **proposed improvement
  (#4), not yet implemented** — `parse.py` does not call `get_toc()` today.
- `print_outline(blocks)` is provided to eyeball the inferred headings before trusting
  downstream chunking/tree-building.

---

### `contract.py` — Output Schema

**Responsibility:** Define the single contract all three approaches must emit.

---

### `verify.py` — Mechanical Quote Verification

**Responsibility:** Validate that every quoted claim exists in source (code-based, not LLM-based).

- **Numeric guard (`_numbers_ok`):** every digit-run asserted in the claim text must
  also appear in the cited quote — so a paraphrased number can't slip through. (This is
  also the source of the known false-positive on incidental numbers like "Section 1" —
  Improvements #2.)
- **Key Feature:** all three approaches call the same `verify_answer`, so trust is
  consistent and lives in code, never in an LLM.

---

### `chunk.py` — Section-Aware Chunking

**Responsibility:** Break blocks into chunks that respect section boundaries.


**Why Section-Aware?** Fixed-size chunking on policy documents fails catastrophically: a chunk containing the tail of "Vacation Policy" + the head of "Sick Leave" produces confidently wrong day counts. Heading-bounded chunks prevent this.

---

### `embeddings.py` — Embedding Provider Abstraction

**Responsibility:** Abstract away embedding provider differences.

Note: methods are `embed_documents` / `embed_query` (not `embed`), and `dim` is an
**attribute, not a method**. There is no separate `OpenAIEmbedder`/`OpenRouterEmbedder`
class — one `OpenAICompatibleEmbedder` handles both, distinguished by base_url + key.

**Config:**
```
EMBEDDING_PROVIDER=local      # offline hashing (default; tests)
EMBEDDING_PROVIDER=openai     # OpenAI /embeddings
EMBEDDING_PROVIDER=openrouter # OpenRouter /embeddings (one key for chat+embed)
```

---

### `keyword.py` — BM25 Keyword Search

**Responsibility:** Lexical search implementation.

Note: the class is `KeywordIndex` (not `BM25Search`); it takes `(id, text)` tuples,
defaults are `k1=1.5, stem=True`, and `search` returns `(id, score)` tuples — not
`Chunk` objects. The analyzer (`analyze`) is the part that determines quality.

**Hybrid Config:**
```
BM25_K1=1.5           # term frequency saturation (default 1.5)
BM25_B=0.75           # length normalization
KEYWORD_STEMMING=true # default true
HYBRID_METHOD=rrf     # or "weighted"
HYBRID_ALPHA=0.5      # weight for dense if weighted
```

---

### `store.py` — In-Memory Vector Store

**Responsibility:** Dense + lexical indexing in RAM (NumPy + BM25).

**Note:** the store exposes the `{dense, lexical}` halves only — there is **no
`search_hybrid` method on the store**. Fusion lives in `retrieve.hybrid_search(store, …)`,
which calls `dense_scored`/`lexical_scored` and combines their rankings:
1. Dense: cosine similarity
2. Lexical: BM25 relevance
3. Fuse: Reciprocal Rank Fusion (no score calibration needed)

---

### `pgstore.py` — Production Vector Store (Postgres + pgvector)

**Responsibility:** Persistent embeddings + corpus in Postgres.


**Note:** same `{dense_scored, lexical_scored, get_text, by_id}` interface as
`InMemoryStore` (so `retrieve.py`/`traditional.py`/`agent.py` are unchanged) — again,
**no `search_hybrid`**. The connection string param is `dsn`, not `database_url`.

**Features:**
- Idempotent ingest (second run reuses table)
- HNSW ANN for fast dense search
- Full-text search (Postgres FTS) for keywords
- Automatic fallback if DB unreachable (when `STORE_BACKEND=auto`)

**Schema (auto-created by `_ensure_schema`, not by hand):**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE pp_chunks (
    id text PRIMARY KEY,
    text text NOT NULL,
    section_path text,
    page_start int, page_end int,
    bboxes jsonb,
    embedding vector(<dim>),   -- <dim> = the embedder's actual size (1536 by default)
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX pp_chunks_tsv ON pp_chunks USING gin(tsv);
-- HNSW index created only when dim ≤ 2000 (pgvector's cap); above that, exact scan:
CREATE INDEX pp_chunks_emb ON pp_chunks USING hnsw (embedding vector_cosine_ops);
```
A `pp_meta` table tracks the dim + corpus signature for idempotent ingest; if the
embedder's dim changes, the table is dropped and rebuilt automatically.

---

### `retrieve.py` — Unified Hybrid Retrieval

**Responsibility:** Abstract dense + lexical search.


Note: the function is `hybrid_search` (not `retrieve_hybrid`); it returns **a list of
IDs**, not `Chunk` objects, and uses `k_each`/`k_final`, not a single `k`.

---

### `tree.py` — Table of Contents Tree

**Responsibility:** Build outline from blocks; tree-search reasoning for PageIndex.


Note the real shapes: fields are `id` / `page_start` / `page_end` / `text` (not
`node_id`/`page_range`/`body_text`); a node's `text` includes **all descendant text**;
and `tree_search` is called `(llm, query, root)` and returns **`(node_ids, trace)`** —
the trace carries the model's "thinking" for the audit log.

**Heading Hierarchy:** inferred from font size (larger size = shallower level).

---

### `generate.py` — Citation-Constrained Generation

**Responsibility:** Invoke LLM with custom JSON schema enforcing citations.


**Prompt Template:**
- Instructs LLM to cite only from provided chunks
- Enforces JSON output schema (claims + quotes)
- Notes that quotes will be verified by code

---

### `traditional.py` — Traditional RAG Orchestration

**Responsibility:** Hybrid retrieve → generate → verify pipeline.

---

### `pageindex.py` — PageIndex RAG Orchestration

**Responsibility:** Tree-search → section load → generate → verify.

---

### `agent.py` — Agentic RAG Tool Loop

**Responsibility:** Bounded LLM agent with tools.

**Tool definitions:** the tool list constant is `TOOLS`; the submit schema is
`CLAIMS_SCHEMA`. (There is no `TOOL_DEFINITIONS`.)

---

### `pipeline.py` — Orchestration Layer

**Responsibility:** Build shared foundation once; route queries.

---

### `llm.py` — Provider Abstraction

**Responsibility:** Unify API differences across Anthropic, OpenAI, OpenRouter.

Note: the interface is `chat(messages, …) -> LLMResponse` (a normalized message/tool
format), **not** `__call__`/`json_mode`. There is no separate `OpenAILLM`/`OpenRouterLLM`
— `OpenAICompatibleLLM` covers both. Sampling params (`temperature`/`top_p`/`seed`)
default to "unset → not sent" so reasoning models that reject them still work.

---

### `config.py` — Configuration Management

**Responsibility:** Load from `.env` and environment variables.

Abridged — the real `Config` has ~25 fields (full list in [config.py](policypal/config.py)).
Note there is **no single `model` field**; each provider has its own:

---

## Docker Deployment

### Services

The real [`docker-compose.yml`](docker-compose.yml) — **no `version:` key** (it's
obsolete in modern Compose and emits a warning if present), all config via
`${VAR:-default}` interpolation, and a health-gated DB dependency. Abridged below
(the full env block is in the file):


### Image Layers

**Dockerfile:**

```dockerfile
FROM python:3.12-slim
# + git (for optional PageIndex clone)
# + pip install -r requirements.txt
# Expose: 8888 (Jupyter), 8501 (Streamlit)
# CMD: jupyterlab --ip=0.0.0.0 --port=8888 ...
```

**Security Notes:**
- JupyterLab runs **token-less** for local development
- Before exposing beyond `localhost`, set `--ServerApp.token` or `--ServerApp.password`
- Postgres runs with default credentials (change in production)

---

## Installation & Setup

### Docker Quick Start

```bash
# 1. Set API key
# Edit .env

# 2. Build & start
docker compose up --build

# 3. Open browser: http://localhost:8888 (Jupyter)

# 4. Inside JupyterLab, run:
open run_eval.ipynb
```

---

## Running the System

### Jupyter Notebooks

1. **`PolicyPal_demo.ipynb`** — Interactive walkthrough of all three approaches
2. **`run_eval.ipynb`** — Evaluation harness with parameter tuning

Inside container:

```bash
docker compose exec jupyter jupyter lab
# http://localhost:8888
```

### Tests

```bash
# Run offline test suite (forces mock provider)
python -m pytest tests/

# Verbose + show print
python -m pytest tests/ -v -s

# Specific test file
python -m pytest tests/test_pipeline.py
```

---

## Evaluation & Results

### Evaluation Framework

**15 Curated Cases:**
- 10 answerable (questions with clear handbook answers)
- 5 adversarial (trick questions, handbook-negative cases)

**Metrics per Approach:**

| Metric | Definition |
|--------|-----------|
| **Abstention** | Did the system correctly say "not in handbook" for unanswerable cases? |
| **Correctness** | Did the answer match the ground truth? (LLM judge) |
| **Evidence Recall** | Were the key facts cited? |
| **Citation Validity** | Do cited quotes actually appear in source? |
| **Tokens / Query** | Average tokens per query (cost proxy) |
| **Calls** | Average LLM calls per query |

### Running Evaluations

```bash
python run_eval.py --approach all --judge --detail
```

**Output format** (the real layout `render_report` prints — one column per approach;
values below are **illustrative placeholders**, not measured results — run the harness
against a real provider for actual numbers):

```
metric                traditional   pageindex     agentic
------------------------------------------------------------
abstention_accuracy     <0–1>         <0–1>         <0–1>
answer_correctness      <0–1>         <0–1>         <0–1>
evidence_recall         <0–1>         <0–1>         <0–1>
citation_validity       <0–1>         <0–1>         <0–1>
faithfulness            <0–1 | n/a>   ...           ...
------------------------------------------------------------
tokens/query            <int>         <int>         <int>
cost/query ($)          <float>       <float>       <float>
latency/query (s)       <float>       <float>       <float>     ← now measured
```

`--detail` adds a per-case line: `abstain✓ correct✓ recall✓ cites 1/1  0.30s`.

### Reproduction notes

- The `mock` provider is deterministic (offline), so the *harness* runs reproducibly —
  but mock metrics reflect plumbing, not model quality, and its latency is local
  compute only (no network).
- Real-provider runs are **not** seeded by default. Set `SEED` (and `TEMPERATURE=0`) in
  `.env` for more repeatable output where the provider supports it; otherwise expect
  run-to-run variation.
- `--detail` prints every case's per-metric result; the aggregate table is always shown.

---

## Performance Breakdown

### Latency

Per-query wall-clock is **measured** (`ans.latency_s`, set in `pipeline.answer()`) and
aggregated by the eval harness as `latency/query (s)`. The figures below are **rough
order-of-magnitude expectations** for a real provider, driven mainly by LLM-call count
(1 / 2 / 3–8) — **not** measured benchmarks. Run `run_eval.py` against your provider to
get real numbers.

| Approach | Expected order of magnitude | Why |
|----------|---|---|
| **Traditional** | ~seconds | 1 LLM call + local retrieve/verify |
| **PageIndex** | ~seconds | 2 LLM calls (tree-search + generate) |
| **Agentic** | several seconds–tens of seconds | 3–8 LLM calls in the tool loop |

> Retrieval, parse, and verify are local and sub-second on a handbook; the LLM call(s)
> dominate. A finer per-stage breakdown is not yet instrumented (Improvements #10).

**Profiling — use the measured field directly:**
```python
ans = pal.answer("How many days of bereavement leave?", approach="traditional")
print(f"Latency: {ans.latency_s:.2f}s")                       # measured wall-clock
print(f"Tokens: {ans.usage['input_tokens']} in, {ans.usage['output_tokens']} out")
print(f"Calls: {ans.usage['calls']}   Cost: ${ans.cost_usd:.4f}")
```

The `demo.py` usage line and the eval table surface this automatically:
`— tokens: 1,707 in + 65 out · 1 LLM call(s) · 0.53s · est. cost $0.0004`.

### Throughput

Single-threaded throughput is roughly the inverse of per-query latency (LLM-call-bound),
so it tracks the ordering above: Traditional > PageIndex > Agentic. Not separately
benchmarked — derive it from the measured `latency/query` for your provider. All three
approaches are stateless after init, so a service can parallelize across queries.

**Parallelization:** All three approaches are stateless after initialization, so embed in a web service with thread/async pooling for higher throughput.

### Storage

**In-Memory (NumPy + BM25):**
- ~1k-5k chunks: <50 MB (embeddings + BM25 index)

**Postgres (pgvector):**
- Dense vectors: 1536 dims × float32 = ~6 KB per chunk (~6 MB for 1k chunks at the default dim)
- Full corpus: chunks are ~450 tokens each; a typical handbook is a few hundred chunks
- Rough order: tens of MB for a typical handbook (dominated by the vectors at high dim)

---

## Future Improvements

### Tier 1 — Correctness & Trust (High Priority)

1. **Handle supersession / effective-dating** (§ guide §1, Improvements #1)
   - Handbook has June-2024 amendment overriding §1
   - When retrieve/cite a section with "supersede", "amendment", or later date, surface both and prefer newer
   - File: `generate.py`, `retrieve.py`, `tree.py`

2. **Refine numeric-claim verification** (Improvements #2)
   - Current: `_numbers_ok` requires all digits to appear in quote
   - False positives: "Section **1**", "June **2024**" flagged incorrectly
   - Fix: Check against union of all citations; ignore section/reference tokens
   - File: `verify.py`

3. **Stronger abstention logic** (Improvements #3)
   - Add explicit "is answer actually in these excerpts?" pre-generation check
   - Run 5 adversarial cases (bereavement, parental leave, 401k, tuition, pets) as regression gate
   - File: `generate.py`

4. **Validate heading extraction** (Improvements #4)
   - Font-size heuristics fail if PDF uses color/numbering
   - Prefer `doc.get_toc()` if available; assert inferred outline non-empty
   - File: `parse.py`, `tree.py`

### Tier 2 — Retrieval Quality (Medium Priority)

5. **Replace local embedder for production** (Improvements #5)
   - Local hashing captures only lexical overlap, not paraphrase
   - Default to `EMBEDDING_PROVIDER=openai` (not local) in non-offline deployments
   - File: `embeddings.py`

6. **Add reranker** (Improvements #6)
   - Cross-encoder or LLM rerank of top ~20 between RRF and generation
   - Lifts precision, especially for near-duplicate policy prose
   - File: `retrieve.py` (new `rerank_step`)

7. **High-dimensional ANN** (Improvements #7b)
   - pgvector HNSW caps at 2000 dims; 4096-dim models fall back to sequential scan
   - Use `halfvec` for ≤4000 dims, or request model's `dimensions` param shortening
   - File: `pgstore.py`

8. **LLM-assisted tree summaries** (Improvements #8)
   - Current: first-N-chars heuristic
   - Replace with batched LLM call for real node summaries
   - Improves PageIndex tree-search significantly
   - File: `tree.py` (`_summarize` function)

### Tier 3 — Evaluation & Ops

9. **Grow eval set & LLM-judge calibration** (Improvements #9)
   - Expand from 15 to ~30 cases (aim for ~25 answerable + ~5 adversarial)
   - Add inter-rater spot-checks for judge faithfulness (position/verbosity biases)
   - Wire into CI as quality gate (citation_validity ≥ 1.0)
   - File: `eval/handbook_cases.json`, `evaluation.py`

10. **Per-query latency & token accounting** (Improvements #10)
    - Record wall-clock latency + tokens per query in `answer.retrieval`
    - Validate cost/latency assumptions (1 vs 1–2 vs 3–8 calls)
    - File: `llm.py`, `pipeline.py`

11. **Caching** (Improvements #11)
    - Cache embeddings (deterministic for `local`)
    - Cache tree-search results per (query, doc-hash)
    - Use Anthropic prompt caching for static handbook context
    - File: `llm.py`, `pipeline.py`

### Tier 4 — Product

12. **Streamlit highlight UI** (Improvements #12)
    - Every citation carries `(page, bbox)`
    - Render cited regions on PDF so reviewer verifies in <15 s
    - Port 8501 already published in `docker-compose.yml`
    - File: new `app.py`

13. **Confidence & contradiction UI** (Improvements #13)
    - When two sources disagree (§1 vs amendment), show both with dates
    - Never silently pick one
    - File: `contract.py` (extend), `app.py`

14. **Router upgrade** (Improvements #14)
    - Current: keyword heuristic (and/or/;)
    - Replace with tiny classifier prompt (detect multi-facet, low-confidence)
    - Fallback: traditional → agentic
    - File: `pipeline.py` (`route` method)

15. **AI-tool governance note** (Improvements #15)
    - Document data egress per approach (see `docs/COMPARISON.md`)
    - Material for HR compliance sign-off
    - File: `docs/GOVERNANCE.md` (new)

---

## Troubleshooting

### Docker Issues

**Port already in use:**

```bash
lsof -i :8888    # find PID
kill <PID>
# or change port in docker-compose.yml
```

**Database won't connect:**

```bash
docker compose logs db
# or check PASSWORD, DATABASE_URL match docker-compose.yml
```

**Rebuild after dependency change:**

```bash
docker compose up --build
```

### Common Runtime Errors

**Provider runs the mock instead of a real model**
- With no API key set, the provider auto-detects to `mock` (offline). Set the matching
  key in `.env` (e.g. `ANTHROPIC_API_KEY`) or set `LLM_PROVIDER` explicitly.
- If you force `LLM_PROVIDER=anthropic` *without* a key, the provider's SDK raises its
  own auth error (not a `KeyError`).

**Embedding dimension changed (e.g. switched models)**
- `pgstore` detects this from the catalog and **drops/rebuilds the table automatically**
  on the next run — no manual step needed.
- To force a clean re-embed/re-ingest for any other reason: `STORE_REBUILD=1`.
- To inspect persisted vectors:
  `docker compose exec db psql -U policypal -d policypal -c "SELECT count(*), vector_dims(embedding) FROM pp_chunks GROUP BY 2;"`

**Headings look wrong / outline is off**
- `parse.py` infers headings purely from font size (no embedded-TOC fallback yet).
  PDFs that signal headings by colour/numbering degrade it.
- Eyeball the inferred headings with `print_outline(pal.blocks)`, or the tree with
  `render_toc(pal.tree)`.

**Quote verification failing on too many claims**
- The fuzzy threshold is the *minimum* match ratio — **higher = stricter**. If valid
  quotes are being rejected, **lower** `fuzz` in `verify.py` (default `0.92`), or check
  that the numeric guard (`_numbers_ok`) isn't flagging incidental numbers (Improvements #2).

### Debugging Queries

**Print retrieval results:**

```python
from policypal.retrieve import hybrid_search

ids = hybrid_search(pal.store, "How many days of vacation?", k_each=20, k_final=6)
for cid in ids:
    c = pal.store.by_id[cid]
    print(f"{c.section_path}: {c.text[:100]}...")
```

**Inspect the audit trail + usage:**

```python
ans = pal.answer("...", approach="traditional")
print(ans.retrieval)          # approach-specific trail: retrieved_ids / node_ids / trace
print(ans.usage, ans.cost_usd, ans.latency_s)
# (Prompts aren't logged by default — add a print in LLM.chat() in llm.py if you need them.)
```

**Profile tree-search** (note arg order `(llm, query, root)`; it returns
`(node_ids, trace)` where node_ids are **strings**):

```python
from policypal.tree import tree_search
node_ids, trace = tree_search(pal.llm, "bereavement", pal.tree)
print(f"Selected node_ids: {node_ids}")
print(f"Reasoning: {trace.get('thinking')}")
```

---

## Appendix: Example Workflows

### Scenario 1: Single-Shot Answering (Company Demo)

```python
from policypal import PolicyPal

pal = PolicyPal("OmniCorp_Handbook.pdf")

questions = [
    "How many days of bereavement leave?",
    "Can I work from home?",
    "What's the 401k match?",
]

for q in questions:
    ans = pal.answer_routed(q)
    print(f"Q: {q}")
    print(f"A: {ans.render()}")
    print(f"Cost: ${ans.cost_usd:.4f}\n")
```

### Scenario 4: Production Deployment

```bash
# Docker Compose
docker compose up --build

# Inside container
docker compose exec jupyter python -c "
from policypal import PolicyPal
pal = PolicyPal('/workspace/handbook.pdf')
ans = pal.answer_routed('Is parental leave available?')
print(ans.render())
"
```

> **No web UI yet.** A Streamlit highlight viewer (`app.py`) is *planned, not
> implemented* — see [Future Improvements](#future-improvements) #12. Port 8501 is
> already published in `docker-compose.yml` so the future app needs no compose change,
> but there is currently no `app.py` to run. Today the interfaces are the CLI

---

## References

- **PDF Parsing:** [PyMuPDF Docs](https://pymupdf.readthedocs.io/)
- **Vector Embeddings:** [pgvector GitHub](https://github.com/pgvector/pgvector)
- **LLM APIs:**
  - [Anthropic API](https://docs.anthropic.com/)
  - [OpenAI API](https://platform.openai.com/docs/)
  - [OpenRouter](https://openrouter.ai/)
- **BM25:** [rank_bm25 Docs](https://github.com/dorianbrown/rank_bm25)

---

**Document Version:** 1.0  
**Last Updated:** June 2024  
**Maintainer:** PolicyPal Development Team

For questions or contributions, please open an issue on the GitHub repository.
