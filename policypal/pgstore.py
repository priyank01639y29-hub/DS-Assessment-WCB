"""Postgres + pgvector store — embeddings and the full corpus live in the DATABASE,
not RAM (guide §2.3).

  * dense search  → pgvector, HNSW index, cosine distance (`<=>`)
  * keyword search → Postgres full-text search (tsvector + GIN, `ts_rank_cd`)

Exposes the same {dense_scored, lexical_scored, get_text, by_id} interface as
InMemoryStore, so retrieve.py / traditional.py / agent.py are unchanged.

Persistence: ingest happens once. If the table already holds the same corpus
(signature + embedding dim match) a later run REUSES it and does not re-embed — that
is the whole point of moving off RAM. Chunk rows are loaded from the DB on demand
(small LRU-free cache), so the resident set stays tiny.

Note on the keyword half: Postgres FTS uses its own 'english' analyzer (Snowball
stemming + stopwords). It is solid, but it does not preserve exact identifiers the way
policypal.keyword's BM25 analyzer does (e.g. POL-853-MEN may be split). If you need
that exact-token behavior, keep STORE_BACKEND=memory, or add a pg_search/BM25 extension.
"""
import hashlib
import re

import numpy as np

from .chunk import Chunk

# Bare alnum terms only — strips any to_tsquery operators so user text can't break it.
_TERM = re.compile(r"[A-Za-z0-9]+")


class PgVectorStore:
    def __init__(self, chunks, embedder, dsn, *, table: str = "pp_chunks",
                 hybrid_method: str = "rrf", hybrid_alpha: float = 0.5,
                 rebuild: bool = False):
        import psycopg
        from pgvector.psycopg import register_vector

        self.embedder = embedder
        # The embedder now reports its REAL dim (probed at init for API models), so trust
        # it; probe directly only as a defensive fallback. A wrong dim would make pgvector
        # reject every insert and silently fall back to RAM.
        self.dim = int(getattr(embedder, "dim", 0) or 0) or self._probe_dim(embedder)
        self.table = table
        self.hybrid_method = hybrid_method
        self.hybrid_alpha = hybrid_alpha

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_schema()
        self.by_id = _DBChunks(self)
        self._ingest(chunks, rebuild)

    def _probe_dim(self, embedder) -> int:
        try:
            return int(np.asarray(embedder.embed_query("dimension probe"),
                                  dtype=np.float32).shape[0])
        except Exception:
            return int(getattr(embedder, "dim", 1536))

    # --- schema ---------------------------------------------------------------
    def _ensure_schema(self):
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pp_meta "
            "(name text PRIMARY KEY, dim int, signature text)")
        # Read the ACTUAL embedding column dimension from the catalog (robust even if
        # pp_meta is stale). format_type renders e.g. 'vector(1536)'.
        row = self.conn.execute(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "JOIN pg_class c ON a.attrelid=c.oid "
            "WHERE c.relname=%s AND a.attname='embedding' AND NOT a.attisdropped",
            (self.table,)).fetchone()
        if row and row[0]:
            m = re.search(r"\((\d+)\)", row[0])
            existing_dim = int(m.group(1)) if m else None
            if existing_dim is not None and existing_dim != self.dim:
                # embedding dim changed → the vector column's fixed width no longer fits
                self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
                self.conn.execute("DELETE FROM pp_meta WHERE name=%s", (self.table,))
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id text PRIMARY KEY,
                text text NOT NULL,
                section_path text,
                page_start int,
                page_end int,
                bboxes jsonb,
                embedding vector({self.dim}),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )""")
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_tsv ON {self.table} USING gin(tsv)")
        # pgvector's HNSW/IVFFlat indexes cap at 2000 dims. Above that we store the
        # vectors but skip the ANN index — exact cosine (sequential scan) is used, which
        # is instant for a handbook. At corpus scale with high-dim models, switch to a
        # halfvec index or reduce the embedding dimension (see docs/IMPROVEMENTS.md).
        if self.dim <= 2000:
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_emb "
                f"ON {self.table} USING hnsw (embedding vector_cosine_ops)")

    # --- ingest (idempotent) --------------------------------------------------
    def _signature(self, chunks) -> str:
        h = hashlib.sha256(str(self.dim).encode())
        for c in chunks:
            h.update(c.id.encode())
            h.update(str(len(c.text)).encode())
        return h.hexdigest()

    def _ingest(self, chunks, rebuild: bool):
        from psycopg.types.json import Jsonb

        sig = self._signature(chunks)
        meta = self.conn.execute(
            "SELECT signature FROM pp_meta WHERE name=%s", (self.table,)).fetchone()
        count = self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
        if count and not rebuild and meta and meta[0] == sig:
            return  # corpus already persisted in the DB — reuse, skip re-embedding

        self.conn.execute(f"TRUNCATE {self.table}")
        vecs = self.embedder.embed_documents([c.text for c in chunks])
        rows = [(c.id, c.text, c.section_path, c.page_start, c.page_end,
                 Jsonb(c.bboxes), np.asarray(v, dtype=np.float32))
                for c, v in zip(chunks, vecs)]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} "
                f"(id, text, section_path, page_start, page_end, bboxes, embedding) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s)", rows)
        self.conn.execute(
            "INSERT INTO pp_meta (name, dim, signature) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET dim=EXCLUDED.dim, signature=EXCLUDED.signature",
            (self.table, self.dim, sig))

    # --- dense / semantic (pgvector) ------------------------------------------
    def dense_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        qv = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        rows = self.conn.execute(
            f"SELECT id, 1 - (embedding <=> %s) FROM {self.table} "
            f"ORDER BY embedding <=> %s LIMIT %s", (qv, qv, k)).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def dense(self, query: str, k: int) -> list[str]:
        return [i for i, _ in self.dense_scored(query, k)]

    # --- lexical / keyword (Postgres FTS) -------------------------------------
    def lexical_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        # plainto_tsquery ANDs every term, which tanks recall on natural-language
        # questions. BM25 is OR-with-ranking, so we OR the terms and rank by ts_rank_cd
        # (only chunks containing >=1 term match; more/rarer matches rank higher).
        terms = _TERM.findall(query)
        if not terms:
            return []
        tsq = " | ".join(dict.fromkeys(t.lower() for t in terms))   # dedup, keep order
        rows = self.conn.execute(
            f"SELECT id, ts_rank_cd(tsv, to_tsquery('english', %s)) AS s "
            f"FROM {self.table} WHERE tsv @@ to_tsquery('english', %s) "
            f"ORDER BY s DESC LIMIT %s", (tsq, tsq, k)).fetchall()
        return [(r[0], float(r[1])) for r in rows if r[1] > 0]

    def lexical(self, query: str, k: int) -> list[str]:
        return [i for i, _ in self.lexical_scored(query, k)]

    # --- corpus access --------------------------------------------------------
    def get_text(self, source_id: str) -> str:
        c = self.by_id.get(source_id)
        return c.text if c else ""

    def _exists(self, cid: str) -> bool:
        return self.conn.execute(
            f"SELECT 1 FROM {self.table} WHERE id=%s", (cid,)).fetchone() is not None

    def _load_chunk(self, cid: str) -> Chunk:
        r = self.conn.execute(
            f"SELECT id, text, page_start, page_end, section_path, bboxes "
            f"FROM {self.table} WHERE id=%s", (cid,)).fetchone()
        if not r:
            raise KeyError(cid)
        return Chunk(id=r[0], text=r[1], page_start=r[2], page_end=r[3],
                     section_path=r[4], bboxes=r[5] or [])


class _DBChunks:
    """Lazy id -> Chunk mapping backed by the DB (keeps the corpus out of RAM)."""

    def __init__(self, store: "PgVectorStore"):
        self._store = store
        self._cache: dict[str, Chunk] = {}

    def __contains__(self, cid): return self._store._exists(cid)

    def __getitem__(self, cid):
        if cid not in self._cache:
            self._cache[cid] = self._store._load_chunk(cid)
        return self._cache[cid]

    def get(self, cid, default=None):
        try:
            return self[cid]
        except KeyError:
            return default
