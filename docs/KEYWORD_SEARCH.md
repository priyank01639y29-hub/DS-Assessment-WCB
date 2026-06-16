# Keyword search & hybrid retrieval (the NotebookLM technique)

## What NotebookLM actually does

NotebookLM grounds every answer in your uploaded sources and cites them. Under the
hood it uses **hybrid search**: a keyword (lexical) retriever running *alongside* a
vector (semantic) retriever, with the two result sets fused before generation. Google
hasn't published the exact internals, but multiple write-ups describe the architecture
as semantic vector search (Vertex AI embeddings) **plus** keyword search (BM25-style),
combined by rank fusion. The keyword half is the part this doc implements and explains.

> The principle is bigger than one product: "the best RAG systems don't choose between
> keyword and vector search — they use both." One commonly cited production result went
> from **62% → 91% retrieval accuracy** just by adding keyword search to a dense-only
> pipeline.

## Why keyword search enhances RAG

Dense/vector search matches *meaning* — great for paraphrase ("my dad died, what am I
entitled to?" → a bereavement section). But it can drift right past **rare, exact
tokens**: identifiers, codes, numbers, acronyms. Those are precisely where keyword
search excels, because BM25 weights rare terms heavily and matches them literally.

Measured on this handbook (`run_eval.py --retrieval`, recall@6):

```
dense (vector)     0.90
keyword (bm25)     1.00   ← rescues exact-term cases: "14 characters", "$1,000", codes
hybrid-rrf         1.00
```

The exact-match cases (password length, dollar thresholds, ref codes) are the ones a
pure vector index drops. With a *strong* semantic embedder the dense number rises and
the two become genuinely complementary — but the structural guarantee holds: hybrid
never trails the better single retriever, and it catches what each alone misses.

## How BM25 keyword scoring works

BM25 is a bag-of-words relevance score, summed over the query terms a document contains:

```
score(d, q) = Σ_t  IDF(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
```

- **IDF(t)** — rarer terms across the corpus score higher (probabilistic IDF). "401k"
  is far more discriminative than "policy".
- **k1** (≈1.2–2.0) — *term-frequency saturation*. The first occurrence of a term
  matters a lot; repeats give diminishing returns, so keyword-stuffed chunks can't run
  away with the score.
- **b** (≈0.75) — *document-length normalization*. `b=0` ignores length; `b=1` fully
  normalizes, so a long chunk doesn't win just by containing more words.

We use `rank_bm25`'s `BM25Okapi` for the math; `k1`/`b` are tunable via env.

## The analyzer is where quality is won or lost

The formula is the easy part. What decides keyword-search quality — and what most
tutorials skip — is the **analyzer**: how raw text becomes tokens. See
[`policypal/keyword.py`](../policypal/keyword.py).

1. **Normalize** — Unicode NFKC + casefold, so "Café"/"café"/"CAFÉ" unify.
2. **Tokenize** — alnum runs that may contain internal `-` `_` `/`, so codes like
   `POL-853-MEN` survive as one token. Thousands separators are stripped so `$1,000`
   → `1000` (one token, not `1` + `000`).
3. **Preserve identifiers/numbers verbatim** — any token with a digit or `-_/` is kept
   exactly: never stop-listed, never stemmed. This is keyword search's whole edge over
   vectors; destroying these tokens throws the advantage away.
4. **Stopword removal** — drop "the/is/what/how…" (a deliberately conservative list —
   words like "many/before/after" are kept because they can carry policy meaning).
5. **Stemming** — `policies → policy`, `passwords → password`. Raises recall, but
   over-stemming hurts precision, so the built-in stemmer is intentionally minimal.

```python
analyze("Ref POL-853-MEN exactly 14 characters $1,000 401k")
# → ['ref', 'pol-853-men', 'exactly', '14', 'character', '1000', '401k']
```

## Fusing keyword + vector scores

Two retrievers produce two ranked lists on **incompatible scales** (BM25 is unbounded;
cosine is [-1, 1]). You cannot just add the raw scores. See
[`policypal/retrieve.py`](../policypal/retrieve.py):

- **RRF (default)** — Reciprocal Rank Fusion combines *rank positions*, not scores, so
  no calibration is needed: `score(d) = Σ 1/(k + rank_d)`. Robust, near parameter-free.
- **Weighted** — min-max normalize each retriever to [0,1], then blend with `alpha`
  (weight on dense). More tunable, but normalization is **mandatory** or one retriever
  silently dominates.

## ⚠️ Special attention (the gotchas)

1. **Index and query must be analyzed identically.** Different tokenization/stemming at
   query time silently tanks recall. (Here both go through the same `analyze()`.)
2. **Don't stem or stop-list identifiers/numbers.** The #1 way to break keyword search
   is to "clean" away the very tokens it's meant to match exactly.
3. **Never blend raw BM25 and cosine scores.** Use RRF, or normalize before weighting.
4. **Stemming is a recall/precision trade.** Aggressive stemming merges distinct words;
   none misses plurals. Tune per corpus; for production prefer a real Snowball/Porter
   stemmer over the minimal built-in one.
5. **Tune k1/b to your chunks.** Short, uniform policy chunks need less length
   normalization than mixed-length documents. `BM25_K1` / `BM25_B` are exposed.
6. **Chunk granularity changes BM25.** IDF and length-normalization are computed over
   *chunks*; very large or very small chunks distort both.
7. **Balance the weighting.** Over-weighting keyword limits semantic recall; over-
   weighting vectors loses exact-match precision. RRF sidesteps this; weighted needs care.
8. **Multilingual / diacritics** need matching normalization (and ideally a language-
   aware analyzer) on both sides.
9. **Postgres FTS: `plainto_tsquery` ANDs every term.** If you back keyword search with
   Postgres full-text search (`pgstore.py`), `plainto_tsquery('english', q)` requires
   *all* query words in a chunk — on natural-language questions that matches almost
   nothing (measured here: recall **0.20**). BM25 is OR-with-ranking; replicate that by
   OR-joining the terms into `to_tsquery('english', 'a | b | c')` and ranking with
   `ts_rank_cd` (recall jumped back to **1.00**). This is the single biggest footgun when
   moving keyword search into the database.
10. **Stack the proven add-ons** the same research highlights: **query expansion/
    rewriting** (LLM generates alternative phrasings to widen lexical recall) and a
    **cross-encoder reranker** after fusion (see [IMPROVEMENTS.md](IMPROVEMENTS.md)).

## How it's wired here

| Piece | Where |
|---|---|
| Analyzer + BM25 `KeywordIndex` | [`policypal/keyword.py`](../policypal/keyword.py) |
| Dense + keyword index (scored) | [`policypal/store.py`](../policypal/store.py) |
| RRF / weighted fusion | [`policypal/retrieve.py`](../policypal/retrieve.py) |
| Retrieval-recall comparison | `evaluate_retrieval()` in [`policypal/evaluation.py`](../policypal/evaluation.py) |

Tunable via env (compose passes them through): `BM25_K1`, `BM25_B`,
`KEYWORD_STEMMING`, `HYBRID_METHOD` (`rrf`|`weighted`), `HYBRID_ALPHA`.

```bash
docker compose exec jupyter python run_eval.py --retrieval          # recall by method
docker compose exec -e HYBRID_METHOD=weighted -e HYBRID_ALPHA=0.3 jupyter \
    python run_eval.py --retrieval                                  # try weighted fusion
```

## Sources

- [RAG with Hybrid Search: How Does Keyword Search Work? — Towards Data Science](https://towardsdatascience.com/rag-with-hybrid-search-how-does-keyword-search-work/)
- [Better RAG Accuracy with Hybrid BM25 + Dense Vector Search — Medium](https://medium.com/@pbronck/better-rag-accuracy-with-hybrid-bm25-dense-vector-search-ea99d48cba93)
- [Full-text search for RAG: BM25 & hybrid search — Redis](https://redis.io/blog/full-text-search-for-rag-the-precision-layer/)
- [Optimizing RAG with Hybrid Search & Reranking — Superlinked VectorHub](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [RAG data ingestion & search techniques — Microsoft Azure AI Search](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/rag-time-journey-2-data-ingestion-and-search-practices-for-the-ultimate-rag-retr/4392157)
- [NotebookLM RAG architecture overview](https://www.scribd.com/document/887551310/NotebookLM-Internal-Framework-Explained)
- [My NotebookLM takeaways from advanced RAG videos — Ethan Lazuk](https://ethanlazuk.com/blog/rag-notebooklm/)
