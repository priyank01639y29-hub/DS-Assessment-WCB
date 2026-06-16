# Suggestions to improve

Concrete, prioritized next steps. Each names the file it touches and why it matters,
grounded in what this handbook actually exposes.

## Tier 1 — correctness & trust (do first)

1. **Handle supersession / effective-dating explicitly.** The handbook's headline trap
   is the June-2024 amendment that overrides §1. Add a post-retrieval step that, when a
   retrieved/cited section contains "supersede", "amendment", "effective", or a later
   date, surfaces *both* and prefers the newer one — and have `generate()` reconcile
   rather than report the stale figure. Today only the agentic path can catch this
   reliably; make it a first-class concern. *(generate.py, retrieve.py, tree.py summaries)*

2. **Refine the numeric-claim guard in `verify.py`.** `_numbers_ok` requires every digit
   in the claim to appear in the quote. Observed false positives with a real LLM: claims
   that mention incidental numbers ("Section **1**", "June **2024**", "W-**2**") get
   flagged UNVERIFIED even when the substantive quote is correct, because those numbers
   aren't in the quote. Fix: check the claim's numbers against the **union** of all its
   citations' quotes, and/or ignore numbers that are part of section/reference tokens.
   Keep the strict check for the *substantive* figures ("14", "$1,000", "10 days").

3. **Stronger abstention.** Add an explicit "is the answer actually in these excerpts?"
   check before generation (or a second-pass self-critique). The eval's 5 adversarials
   (bereavement / parental / 401k / tuition / pets) are the regression guard. *(generate.py)*

4. **Validate heading extraction — the fragile step (guide §1.1).** Font-size heuristics
   fail on PDFs that signal headings by colour/numbering. Prefer `doc.get_toc()` when the
   PDF has an embedded TOC; add a startup assertion that the inferred outline isn't empty
   and roughly matches page count. *(parse.py, tree.py)*

## Tier 2 — retrieval quality

5. **Replace the local hashing embedder for real runs.** It captures lexical overlap, not
   paraphrase. Default `EMBEDDING_PROVIDER=openrouter` (now supported) or `openai` in any
   non-offline deployment; keep `local` only for tests/offline. *(embeddings.py — done; make it the documented default)*

6. **Add a reranker.** A cross-encoder (or an LLM rerank of the top ~20) between RRF and
   generation lifts precision cheaply — especially for the near-duplicate policy prose in
   this handbook. *(new rerank step in retrieve.py)*

7. **~~Port the store to pgvector + `tsvector`.~~ ✅ Done** — `pgstore.py` stores
   embeddings + corpus in Postgres (pgvector HNSW for dense, FTS for keyword), same
   `{dense, lexical}` interface, idempotent ingest. Select with `STORE_BACKEND=pgvector`.
   *Next:* (a) **high-dim ANN** — pgvector's HNSW caps at 2000 dims, so 4096-dim models
   currently fall back to exact (sequential) cosine scan; switch to a `halfvec` HNSW
   index (≤4000 dims) or use a model's `dimensions` param to shorten vectors when the
   corpus grows. (b) ANN tuning (`hnsw.ef_search`) and per-document namespacing for
   multi-doc corpora.

8. **LLM-assisted tree summaries.** `tree._summarize` is currently first-N-chars. A one
   batched LLM call writing real node summaries makes PageIndex tree-search markedly
   better (a summarization error is a retrieval error — guide §3.4). *(tree.py)*

## Tier 3 — evaluation & ops

9. **Grow the eval set & add an LLM-judge calibration.** 15 cases is a smoke set; aim for
   ~25 answerable + ~5 adversarial (guide §5). Add inter-rater spot-checks for the
   `--judge` faithfulness metric (it has position/verbosity/self-preference biases). Wire
   `run_eval.py` into CI as a quality gate (citation_validity must stay 1.00). *(eval/)*

10. **Per-approach latency & token accounting.** Record tokens and wall-clock per query in
    `answer.retrieval`; the cost/latency story (1 vs 1–2 vs 3–8 calls) should be measured,
    not asserted. *(llm.py, pipeline.py)*

11. **Caching.** Cache embeddings (already deterministic for `local`) and tree-search
    results per (query, doc-hash); use Anthropic/OpenRouter prompt caching for the static
    handbook context to cut repeat-query cost. *(llm.py)*

## Tier 4 — product

12. **Streamlit highlight UI (guide §6 step 6).** Every citation already carries
    `(page, bbox)`. Render the cited region on the PDF page so a reviewer verifies in
    <15 s. Port 8501 is already published in compose. *(new app.py)*

13. **Confidence & "show the contradiction" UI.** When two cited sources disagree
    (the §1 vs amendment case), show both with dates rather than silently picking one.

14. **Router upgrade.** `PolicyPal.route` is a keyword heuristic. Replace with a tiny
    classifier prompt that detects multi-facet / cross-reference questions, and fall back
    traditional→agentic on low retrieval confidence. *(pipeline.py)*

15. **AI-tool disclosure / governance note (guide §6 step 7).** Document what egresses to
    which provider per approach (see COMPARISON.md governance row) — material for an HR
    compliance sign-off.
