"""§2.3 Hybrid retrieval — dense (semantic) ⊕ BM25 (keyword), then fused.

HR queries are terminology-dense ("bereavement", "FMLA", "401k vesting") so exact
keyword match is often the strongest signal; dense retrieval covers paraphrase
("my dad died, what am I entitled to?"). This is the NotebookLM-style hybrid: neither
retriever alone is enough, so use both and combine.

Two fusion strategies:
  * RRF (default) — Reciprocal Rank Fusion. Combines *rank positions*, so it needs no
    score calibration between two retrievers whose scores live on different scales
    (BM25 is unbounded; cosine is [-1, 1]). Robust, parameter-light.
  * weighted — min-max normalize each retriever's scores to [0,1], then blend with
    `alpha` (alpha=fraction given to dense). More tunable, but score normalization is
    mandatory or one retriever silently dominates.
"""


def rrf(*rankings, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _minmax(scored: list[tuple[str, float]]) -> dict[str, float]:
    if not scored:
        return {}
    vals = [s for _, s in scored]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {i: (s - lo) / rng for i, s in scored}


def weighted_fusion(dense_scored, lexical_scored, alpha: float = 0.5) -> list[str]:
    d, l = _minmax(dense_scored), _minmax(lexical_scored)
    ids = set(d) | set(l)
    return sorted(ids, key=lambda i: alpha * d.get(i, 0.0) + (1 - alpha) * l.get(i, 0.0),
                  reverse=True)


def hybrid_search(store, query: str, k_each: int = 20, k_final: int = 6,
                  method: str | None = None, alpha: float | None = None) -> list[str]:
    method = method or getattr(store, "hybrid_method", "rrf")
    alpha = alpha if alpha is not None else getattr(store, "hybrid_alpha", 0.5)
    dense = store.dense_scored(query, k_each)
    lexical = store.lexical_scored(query, k_each)
    if method == "weighted":
        fused = weighted_fusion(dense, lexical, alpha)
    else:
        fused = rrf([i for i, _ in dense], [i for i, _ in lexical])
    return fused[:k_final]
