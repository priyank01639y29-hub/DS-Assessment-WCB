"""Environment-driven configuration.

The provider is chosen explicitly via LLM_PROVIDER, or auto-detected from whichever
API key is present. "mock" makes the whole pipeline run offline (no network), which
is what the test suite and the no-key demo use.
"""
import os
from dataclasses import dataclass, fields, replace

# OpenAI embedding model -> dimension (so the in-memory store sizes correctly).
_OPENAI_EMB_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Fallback embedding dimension used when the size is neither set via EMBEDDING_DIM nor
# known from the table above. For API providers the REAL dim is probed from the live
# model at runtime; this is only the fallback if that probe fails.
DEFAULT_EMBEDDING_DIM = 1536


@dataclass
class Config:
    provider: str            # openrouter | anthropic | openai | mock
    anthropic_api_key: str
    anthropic_model: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    # common LLM sampling parameters (None => not sent; provider default applies)
    temperature: float | None
    top_p: float | None
    seed: int | None
    max_tokens: int
    embedding_provider: str  # local | openai | openrouter
    embedding_model: str
    embedding_dim: int            # requested/fallback dim; API real dim is probed at runtime
    embedding_dim_explicit: bool  # True if EMBEDDING_DIM was set (request that exact size)
    database_url: str
    # hybrid keyword+vector retrieval (NotebookLM-style)
    bm25_k1: float
    bm25_b: float
    keyword_stem: bool
    hybrid_method: str       # rrf | weighted
    hybrid_alpha: float      # weight on dense when method=weighted
    # storage backend for embeddings + corpus
    store_backend: str       # auto | memory | pgvector
    store_rebuild: bool      # force re-ingest into pgvector
    # token pricing (USD per 1M tokens) — set to match your model/provider
    cost_input_per_1m: float
    cost_output_per_1m: float


def _opt_float(name: str) -> float | None:
    """Parse an optional float env var. Blank or unset => None (use provider default).

    docker compose passes these as "" when the host var is unset (e.g. TOP_P=${TOP_P:-}),
    so an empty string must count as "not set", not crash on float("").
    """
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


def _opt_int(name: str) -> int | None:
    """Parse an optional int env var. Blank or unset => None (use provider default)."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


def _auto_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def load_config(**overrides) -> Config:
    """Build the config from environment variables, then apply any explicit overrides.

    Every Config field is configurable before the project starts, e.g.:
        load_config(provider="openrouter", embedding_model="qwen/qwen3-embedding-8b",
                    bm25_k1=2.0, store_backend="pgvector", cost_input_per_1m=0.05)
    """
    emb_provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # An empty string counts as unset (docker compose passes EMBEDDING_DIM="" when the
    # host var is unset), so only a non-empty value REQUESTS an explicit size.
    emb_dim_raw = os.getenv("EMBEDDING_DIM", "").strip()
    emb_dim_explicit = bool(emb_dim_raw)
    if emb_provider in ("openai", "openrouter"):
        # OpenRouter ids are prefixed ("openai/text-embedding-3-small"); strip to look up dim.
        # This is only a request/fallback — the real dim is probed from the live model.
        base_model = emb_model.split("/")[-1]
        emb_dim = int(emb_dim_raw) if emb_dim_explicit else _OPENAI_EMB_DIMS.get(base_model, DEFAULT_EMBEDDING_DIM)
    else:
        emb_dim = int(emb_dim_raw) if emb_dim_explicit else DEFAULT_EMBEDDING_DIM   # local hashing embedder
    cfg = Config(
        provider=_auto_provider(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=_opt_float("TEMPERATURE"),
        top_p=_opt_float("TOP_P"),
        seed=_opt_int("SEED"),
        max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
        embedding_provider=emb_provider,
        embedding_model=emb_model,
        embedding_dim=emb_dim,
        embedding_dim_explicit=emb_dim_explicit,
        database_url=os.getenv("DATABASE_URL", ""),
        bm25_k1=float(os.getenv("BM25_K1", "1.5")),
        bm25_b=float(os.getenv("BM25_B", "0.75")),
        keyword_stem=os.getenv("KEYWORD_STEMMING", "1").lower() not in ("0", "false", "no"),
        hybrid_method=os.getenv("HYBRID_METHOD", "rrf").strip().lower(),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.5")),
        store_backend=os.getenv("STORE_BACKEND", "auto").strip().lower(),
        store_rebuild=os.getenv("STORE_REBUILD", "0").lower() not in ("0", "false", "no"),
        cost_input_per_1m=float(os.getenv("COST_INPUT_PER_1M", "0")),
        cost_output_per_1m=float(os.getenv("COST_OUTPUT_PER_1M", "0")),
    )
    if overrides:
        valid = {f.name for f in fields(Config)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"unknown config field(s): {sorted(unknown)}. "
                             f"valid fields: {sorted(valid)}")
        # an explicit embedding_dim override counts as "specified"
        if "embedding_dim" in overrides:
            overrides.setdefault("embedding_dim_explicit", True)
        cfg = replace(cfg, **overrides)
    return cfg


def estimate_cost(usage: dict, config: "Config") -> float:
    """USD cost of an answer's token usage at the configured per-1M-token prices."""
    return (usage.get("input_tokens", 0) / 1e6) * config.cost_input_per_1m + \
           (usage.get("output_tokens", 0) / 1e6) * config.cost_output_per_1m
