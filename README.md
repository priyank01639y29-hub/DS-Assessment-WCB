# PolicyPal — Enterprise RAG with Verifiable Citations

An internal HR assistant that answers employee-handbook questions **and shows exactly
where each answer comes from**, so an HR specialist can confirm it in seconds instead of
re-reading the handbook. Every sentence the system asserts is tied to a verbatim quote
from the source PDF, with section and page, and that quote is checked **mechanically (by
code, not by another LLM)** against the original text before the answer is shown.

> Built for the Data Scientist Challenge: *Enterprise RAG with Verifiable Citations*
> over the OmniCorp Employee Handbook.

---
### Listed Documents

| file | dir | shortcut |
|---|---|---|
|[`README.md`](README.md)|Here you are|High-level summary|
|[`DOCKER.md`](DOCKER.md)|same dir|anything you need to know about the running env|
|[`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md)|same dir|technical summary|
|[`DON'T README.md`](DON'T README.md)|same dir|something beyond technical|
|[`COMPARISON.md`](docs/COMPARISON.md)|/docs|Compare 3 RAGs + KAG and Andrej LLM Wiki|
|[`IMPROVEMENTS.md`](docs/IMPROVEMENTS.md)|/docs|practical guide for further improvements|
|[`KEYWORD_SEARCH.md`](docs/KEYWORD_SEARCH.md)|/docs|considerations for combining keyword + embedding|

---

## The problem this solves

A normal RAG bot says *"You get 20 vacation days"* and the HR specialist still has to go
find that in the handbook — so the automation saves nothing. PolicyPal closes that trust
gap three ways:

1. **Claim-level citations.** The answer is decomposed into individual claims, each with
   its own citation (section heading + page + exact quote). HR verifies a single sentence
   at a time, not a paragraph.
2. **Mechanical verification.** [`policypal/verify.py`](policypal/verify.py) re-locates
   every quoted span in the parsed PDF. If a quote can't be found verbatim, the claim is
   flagged ⚠️ rather than shown as trusted. This is deterministic code, so the check
   itself never hallucinates.
3. **An audit trail.** Each answer carries the retrieval trace (which sections/chunks were
   read, which tools the agent called), so a reviewer can see *how* the answer was reached.

It also handles the adversarial cases that erode trust: when the handbook has a later
amendment that supersedes an earlier section, or when the answer simply isn't in the
document, the system is expected to reflect the amendment or **abstain** rather than
guess.

---

## Solution at a glance

One shared **parse → cite → verify** layer feeds three swappable retrieval strategies, so
they can be compared on the same footing:

| Approach | How it retrieves | Best for |
|---|---|---|
| **Traditional** | Hybrid search — dense embeddings + BM25 keyword, fused with Reciprocal Rank Fusion | Simple, fast lookups |
| **PageIndex** | Reasoning tree-search over the document outline (no embeddings) | Structured docs, offline |
| **Agentic** | Tool loop (search → browse → read → verify), self-correcting | Complex multi-part questions |

A router (`pal.answer_routed`) auto-picks: multi-part questions go to the agent, the rest
to PageIndex with a hybrid-RAG fallback. See
[`docs/COMPARISON.md`](docs/COMPARISON.md) for why three, and
[`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md) for the speed/cost/quality trade-offs.

---

## Where to start (the notebooks)

The solution is orchestrated from Jupyter. Two notebooks, both submitted **with their
output cells included**:

- **[`PolicyPal_demo.ipynb`](PolicyPal_demo.ipynb) — the main entry point.** Runs the
  end-to-end workflow: parse the handbook, inspect the inferred outline and the PageIndex
  tree, ask an answerable question through all three approaches (showing the verified
  claims ✅/⚠️ with citations), then run the two trust-critical edge cases — a policy
  superseded by a later amendment, and a question not in the handbook (abstain) — and
  finally the audit trail.
- **[`run_eval.ipynb`](run_eval.ipynb) — evaluation.** Runs the curated test set through
  every approach and reports retrieval recall, answer correctness, abstention on
  unanswerable questions, citation validity, latency, and token cost. All run parameters
  are set in its first config cell.

---

## Running it

The whole thing runs in Docker (JupyterLab + a pgvector-enabled Postgres). Full details
in [`DOCKER.md`](DOCKER.md); the short version:

```bash
docker compose up --build      # builds the image, starts JupyterLab + Postgres
```

Then open **http://localhost:8888** and run
[`PolicyPal_demo.ipynb`](PolicyPal_demo.ipynb) top to bottom.

### Configure your own keys and model — edit `.env`

This repo ships a `.env` file that the Docker stack loads automatically. **Open it and
put in your own values** before running:

- `OPENROUTER_API_KEY` — your key (or switch `LLM_PROVIDER` to `anthropic`/`openai` and
  set the matching `*_API_KEY`).
- `OPENROUTER_MODEL` / `EMBEDDING_MODEL` — whichever chat + embedding models you want to
  evaluate.
- `COST_INPUT_PER_1M` / `COST_OUTPUT_PER_1M` — your model's price, so the token-cost
  estimates match your provider.

With **no key set**, the package falls back to `LLM_PROVIDER=mock` and runs the full
pipeline offline (deterministic, quote-verified). The mock is for smoke-testing the
plumbing — it doesn't reason, so set a real provider in `.env` for meaningful answers and
eval numbers.

---

## Repository layout

```
README.md                 ← you are here (submission overview)
PolicyPal_demo.ipynb      ← main notebook: end-to-end demo
run_eval.ipynb            ← evaluation notebook
policypal/                ← the package (parse, retrieve, generate, verify, 3 approaches)
  README.md               ← module-by-module map
eval/handbook_cases.json  ← curated test set (answerable + adversarial cases)
tests/                    ← offline pytest suite (mock provider)
docs/                     ← architecture, comparison, keyword-search, improvements
DOCKER.md                 ← environment setup
TECHNICAL_GUIDE.md        ← deep dive
.env                      ← runtime config — set your own key/model/specs here
```

---

## Reference: use of AI tools

Per the challenge's disclosure requirement.

### Tools used and what each was for

| Tool | Used for |
|---|---|
| **Claude (Anthropic), via the Claude Code CLI** | The primary development assistant — architecture brainstorming (the three-approach design and the shared parse→cite→verify layer), code generation across the `policypal` package, debugging, writing the test suite and the curated eval cases, and drafting documentation. |
| **Runtime LLM (configurable via `.env`)** | The model the system itself calls at answer time — for generation, for the PageIndex tree-search reasoning, and for the agentic tool loop. Configured through `LLM_PROVIDER` / `OPENROUTER_MODEL` (OpenRouter, Anthropic, or OpenAI). This is part of the product, not a build tool. |

### Known limitations, assumptions, and concerns

- **AI-generated code was reviewed, but not line-by-line exhaustively.** The deterministic
  parts (PDF parsing, the mechanical quote verifier, retrieval fusion) are the trust
  anchors and are covered by tests; the LLM-driven parts are inherently
  non-deterministic.
- **The verifier checks *quotes*, not *reasoning*.** It guarantees every cited quote
  really appears in the handbook, which is the core trust requirement. It does **not**
  guarantee the model selected the *right* passage or interpreted it correctly — that's
  what the eval set measures, and why claim-level citations exist (so a human catches a
  wrong-but-real quote quickly).
- **The eval "ground truth" is curated, partly with AI help,** so it reflects the
  authors' reading of the handbook and a finite set of phrasings. Treat the metrics as
  directional, not absolute.
- **Retrieval quality depends on the embedding provider.** The offline `local` embedder is
  a lexical-ish hashing fallback; real paraphrase coverage needs a hosted embedding model
  (set `EMBEDDING_PROVIDER`). The BM25 keyword half backstops jargon either way.
- **Tree-node summaries are heuristic** (first sentence) to keep index-building offline;
  swap in an LLM call for higher-quality summaries.
- **Answer quality, latency, and cost vary with the model you configure.** The shipped
  `.env` values are examples; set your own provider, model, and pricing.
- **no Cross-reference performance Test** which is crucial for RAG.

---

## Security note for the evaluator

The notebook server runs without an auth token for local convenience, and the `.env` is
loaded automatically. Replace the API key with your own and keep `.env` out of any further
sharing — it is not meant to leave your machine.
