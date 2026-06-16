# `policypal` — three RAG architectures over an employee handbook

Implements the design in [`../rag_implementation_guide.md`](../rag_implementation_guide.md):
one shared parse/citation/verification layer, three swappable retrieval strategies.

```python
from policypal import PolicyPal

pal = PolicyPal("OmniCorp_Handbook_Challenge V2.pdf")
ans = pal.answer("How many days of bereavement leave do I get?", approach="pageindex")
print(ans.render())          # claims with ✅/⚠️ verification flags + section/page citations
```

`approach` ∈ `{"traditional", "pageindex", "agentic"}`, or use `pal.answer_routed(q)`
to auto-pick (multi-part → agentic, else PageIndex with a hybrid-RAG fallback).

## Module map

| File | Guide § | Role |
|---|---|---|
| `parse.py` | 1.1 | PyMuPDF → `Block`s with page/bbox/heading (shared provenance) |
| `contract.py` | 1.2 | `Citation` / `Claim` / `Answer` — the one output schema |
| `verify.py` | 1.3 | **Mechanical** quote verification (code, not an LLM) |
| `chunk.py` | 2.2 | Heading-bounded chunks (never straddle two policies) |
| `embeddings.py` | — | Local hashing embedder (offline) / OpenAI / OpenRouter |
| `keyword.py` | 2.3 | BM25 keyword search + analyzer (the NotebookLM hybrid half) |
| `store.py` | 2.3 | In-RAM dense (NumPy) + lexical (BM25) index |
| `pgstore.py` | 2.3 | **pgvector + Postgres FTS store — embeddings in the DB** |
| `retrieve.py` | 2.3 | Hybrid search fused with Reciprocal Rank Fusion |
| `generate.py` | 2.4 | Citation-constrained JSON generation (shared) |
| `traditional.py` | 2 | **Approach 1** — hybrid retrieve → generate → verify |
| `tree.py` | 3.2–3.3 | PageIndex tree build + reasoning tree-search |
| `pageindex.py` | 3 | **Approach 2** — tree-search → load section → generate → verify |
| `agent.py` | 4 | **Approach 3** — tool loop (search/browse/read/verify) |
| `pipeline.py` | 5 | `PolicyPal`: builds the shared layer once, routes queries |
| `llm.py` | 0 | Provider abstraction: OpenRouter / Anthropic / OpenAI / mock |

## Providers

Set `LLM_PROVIDER` (or let it auto-detect from whichever key is present):

| Provider | Chat | Embeddings |
|---|---|---|
| `openrouter` | ✅ via OpenAI-compatible gateway | ✅ |
| `anthropic` | ✅ | ❌ (use `local` or OpenAI) |
| `openai` | ✅ | ✅ |
| `mock` | ✅ offline, deterministic, quote-verified | `local` |

`mock` runs the entire pipeline with no API key — it quotes real source text so the
mechanical verifier passes. That's what `pytest` and the no-key `demo.py` use.

## Notes / next steps

- **Two storage backends** (`STORE_BACKEND`): `memory` (NumPy + BM25, in RAM) or
  `pgvector` (embeddings + corpus persisted in Postgres, dense search via HNSW, keyword
  via Postgres FTS). `auto` (default) uses pgvector when `DATABASE_URL` is set. See
  `pgstore.py`; ingest is idempotent (a second run reuses the table, no re-embedding).
- **Local embeddings are lexical-ish** (hashing). For real paraphrase coverage set
  `EMBEDDING_PROVIDER=openai`. The BM25 half of the hybrid backstops jargon either way.
- **Tree summaries are heuristic** (first sentence) to keep index-building offline;
  swap an LLM call into `tree._summarize` for higher-quality node summaries.
