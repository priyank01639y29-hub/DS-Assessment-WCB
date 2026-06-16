# PolicyPal — Docker environment

A reproducible Jupyter environment with every package from
`rag_implementation_guide.md`, plus a pgvector-enabled Postgres for the
traditional-RAG path (§2.3).

## What's inside

| Service   | Image / build        | Purpose                                                        |
|-----------|----------------------|----------------------------------------------------------------|
| `jupyter` | built from `Dockerfile` | JupyterLab + all guide packages; mounts this folder at `/workspace` |
| `db`      | `pgvector/pgvector:pg16` | Postgres 16 with the `vector` extension preinstalled         |

Python packages (see `requirements.txt`): `pymupdf`, `anthropic`, `openai`,
`pgvector`, `psycopg[binary]`, `rank_bm25`, `numpy`, `streamlit`, `pytest`, plus
`jupyterlab` / `notebook`.

## Quick start

```bash
docker compose up --build   # first run pulls images + installs packages
docker compose down  # end docker env
docker compose down -v   # wipte out the vectors of pgvector
```

Then open **http://localhost:8888** — no token/password (local dev default).
The handbook PDFs in this folder appear under `/workspace` inside the notebook.
Open `PolicyPal_demo.ipynb` to drive the three RAG approaches.

## Run the RAG pipeline (`policypal` package)

open **`run_eval.ipynb`** in JupyterLab for the same evaluation as a notebook — it
shows every question, its answer, and the per-question result, plus the aggregate
quality/cost table. All run + model + retrieval + storage + cost parameters are set in
its first config cell (via `load_config(**overrides)`) before the run.

The `policypal/` package implements the three architectures from
`rag_implementation_guide.md` (traditional / PageIndex / agentic) over a shared
parse→cite→verify layer. See `policypal/README.md`. With no API key it runs the
`mock` provider fully offline.

Docs:
[`docs/COMPARISON.md`](docs/COMPARISON.md) (RAG types vs. direct file upload),
[`docs/KEYWORD_SEARCH.md`](docs/KEYWORD_SEARCH.md) (hybrid keyword+vector search, the
NotebookLM technique), [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) (next steps). The
curated eval set is `eval/handbook_cases.json` (10 answerable + 5 adversarial).

Hybrid-search tuning env vars: `BM25_K1`, `BM25_B`, `KEYWORD_STEMMING`,
`HYBRID_METHOD` (rrf|weighted), `HYBRID_ALPHA`.

### Token count & estimated cost

Every answer reports tokens and an estimated cost, e.g.
`— tokens: 1,523 in + 81 out · 1 LLM call(s) · est. cost $0.0058`. `run_eval.py` adds
`tokens/query` and `cost/query` rows per approach. Set your model's pricing in `.env`
(USD per 1M tokens): `COST_INPUT_PER_1M`, `COST_OUTPUT_PER_1M`.

### LLM providers

`LLM_PROVIDER` selects the backend (auto-detected from whichever key is set):

| value | uses |
|---|---|
| `openrouter` | OpenAI-compatible gateway — `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| `anthropic`  | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `openai`     | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| `mock`       | offline, deterministic, no key |

OpenRouter has no embeddings endpoint, so dense retrieval uses `EMBEDDING_PROVIDER=local`
(offline hashing) unless you set `EMBEDDING_PROVIDER=openai` with an `OPENAI_API_KEY`.

Stop with `Ctrl-C`, or `docker compose down` (add `-v` to also wipe the Postgres
volume).

## Storage: embeddings in Postgres (pgvector)

By default (`STORE_BACKEND=auto`, with `DATABASE_URL` set) PolicyPal stores embeddings
and the full corpus **in Postgres** — dense search runs in pgvector (HNSW, cosine),
keyword search in Postgres full-text search. Nothing retrieval-related stays in RAM, and
ingest is idempotent (a second run reuses the table, no re-embedding).

```bash
# force in-RAM instead (NumPy + BM25):
docker compose exec -e STORE_BACKEND=memory jupyter python demo.py
# force re-embed + re-ingest (e.g. after changing chunking or the embedding model):
docker compose exec -e STORE_REBUILD=1 jupyter python demo.py

# inspect the persisted vectors:
docker compose exec db psql -U policypal -d policypal \
  -c "SELECT count(*), vector_dims(embedding) FROM pp_chunks GROUP BY 2;"
```

`STORE_BACKEND` = `auto` | `memory` | `pgvector`. With `pgvector`, failures are strict;
with `auto`, an unreachable DB falls back to in-RAM with a warning.

## Using Postgres from a notebook

The `db` service is reachable at the env var `DATABASE_URL`:

```python
import os, psycopg
conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")   # §2.3
```

Don't need the database (in-memory NumPy + `rank_bm25` prototype only)? Run just
the notebook service: `docker compose up --build jupyter`.


## PageIndex (§3.2 Option A)

`git` is available in the image if you want the upstream tree builder:

```bash
git clone https://github.com/VectifyAI/PageIndex && cd PageIndex
pip install -r requirements.txt
```

## Notes

- **Security:** the notebook server runs with no auth token for local
  convenience. Before exposing it beyond `localhost`, set a token/password in
  the `CMD` in `Dockerfile`.
- **Adding packages:** edit `requirements.txt`, then `docker compose up --build`.
- **Rebuild after dependency changes only** — code/notebook edits are live via
  the bind mount, no rebuild needed.
