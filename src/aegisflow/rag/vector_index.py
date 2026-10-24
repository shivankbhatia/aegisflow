"""PostgreSQL/pgvector storage and cosine-similarity retrieval."""

from __future__ import annotations

import asyncio
import hashlib
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
    PRIMARY KEY (id, strategy)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_strategy_source_idx
    ON chunks (strategy, source);
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

    def explain_search(
        self,
        query_embedding: Sequence[float],
        *,
        strategy: str,
        source: str | None = None,
        k: int = 5,
    ) -> list[str]:
        """Return the planner output for the filtered nearest-neighbor query."""
        sql = (
            "EXPLAIN SELECT id FROM chunks WHERE strategy = %s "
        )
        params: list[object] = [strategy]
        if source is not None:
            sql += "AND source = %s "
            params.append(source)
        sql += "ORDER BY embedding <=> %s LIMIT %s"
        params.extend([self._vector(query_embedding), k])
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [row[0] for row in rows]


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
