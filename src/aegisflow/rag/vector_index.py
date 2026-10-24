"""PostgreSQL/pgvector storage and cosine-similarity retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from typing import Awaitable

from aegisflow.rag.types import Chunk


CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id text NOT NULL,
    strategy text NOT NULL,
    doc_id text NOT NULL,
    source text NOT NULL,
    heading_path text[] NOT NULL,
    text text NOT NULL,
    token_count int NOT NULL,
    embedding vector(384) NOT NULL,
    content_hash text NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    PRIMARY KEY (id, strategy)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_strategy_source_idx
    ON chunks (strategy, source);
"""

MIGRATE_SPARSE_SQL = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_search_vector_gin_idx
    ON chunks USING gin (search_vector);
"""

UPSERT_SQL = """
INSERT INTO chunks
    (id, strategy, doc_id, source, heading_path, text, token_count, embedding, content_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id, strategy) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    source = EXCLUDED.source,
    heading_path = EXCLUDED.heading_path,
    text = EXCLUDED.text,
    token_count = EXCLUDED.token_count,
    embedding = EXCLUDED.embedding,
    content_hash = EXCLUDED.content_hash
WHERE chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash
"""


class PgVectorIndex:
    """Open short-lived psycopg connections for safe concurrent retrieval."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        connection = psycopg.connect(self.dsn)
        register_vector(connection)
        return connection

    @staticmethod
    def _vector(values: Sequence[float]):
        from pgvector import Vector

        return Vector(list(values))

    def initialize(self) -> None:
        """Create the vector extension, table, and retrieval indexes."""
        import psycopg

        # The adapter cannot resolve the vector OID until the extension exists.
        with psycopg.connect(self.dsn) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        with self._connect() as connection:
            connection.execute(CREATE_SCHEMA_SQL)
            connection.execute(MIGRATE_SPARSE_SQL)

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        batch_size: int = 256,
    ) -> None:
        """Insert or update chunks in batches; hashes avoid unchanged updates."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows = self._rows(chunks, embeddings)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for offset in range(0, len(rows), batch_size):
                    cursor.executemany(UPSERT_SQL, rows[offset : offset + batch_size])

    def replace_strategy(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        strategy: str,
        doc_ids: Sequence[str],
        batch_size: int = 256,
    ) -> None:
        """Upsert a complete strategy snapshot and remove stale chunk IDs.

        Chunk writes and stale-row deletion share one transaction, so readers
        see either the old snapshot or the replacement snapshot.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if any(c.metadata.get("chunk_strategy") != strategy for c in chunks):
            raise ValueError("every chunk must match the requested strategy")
        if any(c.doc_id not in doc_ids for c in chunks):
            raise ValueError("every chunk document must be included in doc_ids")
        rows = self._rows(chunks, embeddings)
        ids = [chunk.id for chunk in chunks]
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for offset in range(0, len(rows), batch_size):
                    cursor.executemany(UPSERT_SQL, rows[offset : offset + batch_size])
                if doc_ids:
                    cursor.execute(
                        "DELETE FROM chunks WHERE strategy = %s AND doc_id = ANY(%s) "
                        "AND id <> ALL(%s)",
                        (strategy, list(doc_ids), ids),
                    )

    def _rows(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        rows = []
        for chunk, vector in zip(chunks, embeddings):
            if len(vector) != 384:
                raise ValueError("pgvector index requires 384-dimensional embeddings")
            metadata = chunk.metadata
            text = chunk.text
            rows.append(
                (
                    chunk.id,
                    metadata["chunk_strategy"],
                    chunk.doc_id,
                    metadata["source"],
                    metadata.get("heading_path", []),
                    text,
                    int(metadata["token_count"]),
                    self._vector(vector),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
        return rows

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        strategy: str,
        k: int = 5,
        source: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return ranked ``(chunk_id, cosine_similarity)`` pairs."""
        if len(query_embedding) != 384:
            raise ValueError("pgvector index requires a 384-dimensional query embedding")
        if k < 1:
            raise ValueError("k must be positive")
        sql = (
            "SELECT id, 1 - (embedding <=> %s) AS score FROM chunks "
            "WHERE strategy = %s"
        )
        vector = self._vector(query_embedding)
        params: list[object] = [vector, strategy]
        if source is not None:
            sql += " AND source = %s"
            params.append(source)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([vector, k])
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def count(self, strategy: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE strategy = %s", (strategy,)
            ).fetchone()
        return int(row[0])

    def fetch_texts(self, ids: Sequence[str], *, strategy: str) -> dict[str, str]:
        """Fetch passage text for reranking without loading the whole corpus."""
        if not ids:
            return {}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text FROM chunks WHERE strategy = %s AND id = ANY(%s)",
                (strategy, list(ids)),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def sparse_search(
        self,
        query: str,
        *,
        strategy: str,
        k: int = 50,
        source: str | None = None,
    ) -> list[tuple[str, float]]:
        """Rank simple-config PostgreSQL full-text matches with ``ts_rank_cd``.

        This is PostgreSQL full-text ranking, not BM25.
        """
        if k < 1:
            raise ValueError("k must be positive")
        # PostgreSQL's parser indexes ``add_task`` as two lexemes. Replace
        # underscores with spaces before websearch parsing so it does not turn
        # the identifier into the phrase query ``'add' <-> 'task'``.
        normalized_query = re.sub(r"_+", " ", query)
        sql = (
            "SELECT id, ts_rank_cd(search_vector, websearch_to_tsquery('simple', %s)) AS score "
            "FROM chunks WHERE strategy = %s "
            "AND search_vector @@ websearch_to_tsquery('simple', %s)"
        )
        params: list[object] = [normalized_query, strategy, normalized_query]
        if source is not None:
            sql += " AND source = %s"
            params.append(source)
        sql += " ORDER BY score DESC, id LIMIT %s"
        params.append(k)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
            if not rows:
                # websearch's implicit AND can miss relevant chunks when a
                # code sample was included externally and is absent from the
                # indexed page. Fall back to OR across the parsed terms.
                stopwords = {
                    "a", "about", "after", "all", "also", "an", "and", "are", "as",
                    "at", "be", "can", "do", "does", "for", "from", "how", "i",
                    "in", "is", "it", "of", "on", "or", "the", "to", "use",
                    "what", "when", "where", "which", "who", "why", "with",
                    "you", "your",
                }
                terms = [
                    term for term in re.findall(r"[A-Za-z0-9]+", normalized_query)
                    if len(term) > 1 and term.lower() not in stopwords
                ][:8]
                if len(terms) > 1:
                    fallback_query = " OR ".join(terms)
                    fallback_params = [fallback_query, strategy, fallback_query]
                    if source is not None:
                        fallback_params.append(source)
                    fallback_params.append(k)
                    rows = connection.execute(sql, fallback_params).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def explain_search(
        self,
        query_embedding: Sequence[float],
        *,
        strategy: str,
        source: str | None = None,
        k: int = 5,
    ) -> list[str]:
        """Return the planner output for the filtered nearest-neighbor query."""
        sql = "EXPLAIN SELECT id FROM chunks WHERE strategy = %s "
        params: list[object] = [strategy]
        if source is not None:
            sql += "AND source = %s "
            params.append(source)
        sql += "ORDER BY embedding <=> %s LIMIT %s"
        params.extend([self._vector(query_embedding), k])
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [row[0] for row in rows]


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[str],
    k_rrf: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists using weighted reciprocal rank fusion.

    Ranks are one-based. Each ID contributes ``weight / (k_rrf + rank)``
    once per input list; first occurrence wins if a list contains duplicates.
    """
    if k_rrf < 1:
        raise ValueError("k_rrf must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one value per ranked list")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    scores: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights):
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ids, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k_rrf + rank)
    # Python's sort is stable, so equal scores retain first-list (dense) order.
    fused = sorted(scores.items(), key=lambda item: -item[1])
    return fused[:limit] if limit is not None else fused

def make_retriever(
    index: PgVectorIndex,
    embed_query: Callable[[str], Sequence[float]],
    *,
    strategy: str,
    source: str | None = None,
) -> Callable[[str, int], Awaitable[list[str]]]:
    """Adapt dense search to the evaluation harness's async ``(query, k)`` API."""
    async def retrieve(query: str, k: int) -> list[str]:
        def run_search() -> list[tuple[str, float]]:
            vector = embed_query(query)
            return index.search(vector, strategy=strategy, k=k, source=source)

        ranked = await asyncio.to_thread(run_search)
        return [chunk_id for chunk_id, _score in ranked]

    return retrieve


def make_hybrid_retriever(
    index: PgVectorIndex,
    embed_query: Callable[[str], Sequence[float]],
    *,
    strategy: str,
    source: str | None = None,
    dense_k: int = 50,
    sparse_k: int = 50,
    k_rrf: int = 60,
    weights: Sequence[float] = (1.0, 0.5),
) -> Callable[[str, int], Awaitable[list[str]]]:
    """Adapt dense+sparse retrieval to the async evaluation harness signature."""
    if dense_k < 1 or sparse_k < 1:
        raise ValueError("dense_k and sparse_k must be positive")
    if len(weights) != 2:
        raise ValueError("hybrid weights must contain dense and sparse weights")

    async def retrieve(query: str, k: int) -> list[str]:
        def run_search() -> list[str]:
            vector = embed_query(query)
            dense = index.search(vector, strategy=strategy, k=dense_k, source=source)
            sparse = index.sparse_search(query, strategy=strategy, k=sparse_k, source=source)
            fused = reciprocal_rank_fusion(
                [chunk_id for chunk_id, _ in dense],
                [chunk_id for chunk_id, _ in sparse],
                k_rrf=k_rrf,
                weights=weights,
                limit=k,
            )
            return [chunk_id for chunk_id, _ in fused]

        return await asyncio.to_thread(run_search)

    return retrieve
