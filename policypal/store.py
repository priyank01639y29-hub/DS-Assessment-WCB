"""In-memory hybrid index: dense vectors (NumPy) + lexical BM25 (keyword.py).

The guide's production target is pgvector + Postgres tsvector (§2.3); this in-memory
store is the sanctioned prototype path — it needs no database, runs in one process,
and exposes the {dense, lexical} interface the hybrid retriever (§2.3) expects.

Both retrievers expose a *_scored variant returning (id, score) so the fusion layer
(retrieve.py) can do either rank fusion (RRF) or normalized weighted fusion.
"""
import numpy as np

from .chunk import Chunk
from .keyword import KeywordIndex


class InMemoryStore:
    def __init__(self, chunks: list[Chunk], embedder, k1: float = 1.5, b: float = 0.75,
                 stem: bool = True, hybrid_method: str = "rrf", hybrid_alpha: float = 0.5):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.embedder = embedder
        self.matrix = embedder.embed_documents([c.text for c in chunks])  # (N, D), L2-normed
        self.keyword = KeywordIndex([(c.id, c.text) for c in chunks], k1=k1, b=b, stem=stem)
        self.hybrid_method = hybrid_method
        self.hybrid_alpha = hybrid_alpha

    # --- dense / semantic ---
    def dense_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        sims = self.matrix @ q                        # cosine (both L2-normalized)
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i].id, float(sims[i])) for i in order]

    def dense(self, query: str, k: int) -> list[str]:
        return [i for i, _ in self.dense_scored(query, k)]

    # --- lexical / keyword (BM25) ---
    def lexical_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.keyword.search(query, k)

    def lexical(self, query: str, k: int) -> list[str]:
        return [i for i, _ in self.lexical_scored(query, k)]

    def get_text(self, source_id: str) -> str:
        c = self.by_id.get(source_id)
        return c.text if c else ""
