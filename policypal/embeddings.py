"""Dense embeddings for the traditional-RAG path.

Three backends:
  * local      — dependency-free hashing embedder. Deterministic, offline, prototype
                 quality. Captures lexical overlap (good for HR jargon); weak on pure
                 paraphrase — the BM25+RRF hybrid backstops that.
  * openai     — real embeddings (text-embedding-3-*) via api.openai.com.
  * openrouter — real embeddings via OpenRouter's OpenAI-compatible /embeddings
                 endpoint (one key for chat + embeddings). Model ids are prefixed,
                 e.g. "openai/text-embedding-3-small".

All vectors are L2-normalized so a dot product is cosine similarity.
"""
import hashlib

import numpy as np

from .config import Config, load_config
from .util import tokenize


def _stable_hash(tok: str) -> int:
    """Deterministic across processes (unlike built-in hash(), which is salted)."""
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(),
                          "big")


def make_embedder(config: Config | None = None):
    config = config or load_config()
    p = config.embedding_provider
    # If EMBEDDING_DIM was set explicitly, request that exact size from the API; otherwise
    # leave it None so the embedder probes whatever the model actually returns.
    requested = config.embedding_dim if config.embedding_dim_explicit else None
    if p == "openai":
        return OpenAICompatibleEmbedder(
            base_url=config.openai_base_url, api_key=config.openai_api_key,
            model=config.embedding_model, dim=config.embedding_dim, requested_dim=requested)
    if p == "openrouter":
        return OpenAICompatibleEmbedder(
            base_url=config.openrouter_base_url, api_key=config.openrouter_api_key,
            model=config.embedding_model, dim=config.embedding_dim, requested_dim=requested,
            default_headers={"HTTP-Referer": "https://github.com/policypal",
                             "X-Title": "PolicyPal"})
    return HashingEmbedder(config.embedding_dim)


class HashingEmbedder:
    """Feature-hashing bag-of-words with signed buckets + sublinear tf weighting."""

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = _stable_hash(tok)
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0   # signed hashing
        nz = v != 0
        v[nz] = np.sign(v[nz]) * (1.0 + np.log(np.abs(v[nz])))   # damp repeats
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else \
            np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class OpenAICompatibleEmbedder:
    """Works against any OpenAI-schema /embeddings endpoint (OpenAI or OpenRouter).

    `self.dim` is authoritative — it reflects the size the model ACTUALLY returns, not a
    name-based guess:
      * requested_dim set (explicit EMBEDDING_DIM) → ask the API for that size (Matryoshka
        models like text-embedding-3 / qwen3-embedding honor `dimensions`); dim = that.
      * otherwise → probe the live model once and use whatever it returns.
      * if the probe fails (offline/no key) → fall back to `dim` (default 1536).
    """

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1536,
                 requested_dim: int | None = None, default_headers: dict | None = None):
        from openai import OpenAI

        self.model = model
        self.requested_dim = requested_dim
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             default_headers=default_headers or {})
        self.dim = requested_dim or self._probe_dim(fallback=dim)

    def _extra(self) -> dict:
        # only pass `dimensions` when the caller explicitly requested a size
        return {"dimensions": self.requested_dim} if self.requested_dim else {}

    def _probe_dim(self, fallback: int) -> int:
        try:
            r = self.client.embeddings.create(model=self.model, input=["dimension probe"])
            return len(r.data[0].embedding)
        except Exception:
            return fallback

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts, **self._extra())
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 100):          # batch to stay under limits
            out.append(self._embed(texts[i : i + 100]))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]
