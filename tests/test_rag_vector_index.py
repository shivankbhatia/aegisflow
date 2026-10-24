import os
import time
import uuid

import psycopg
import pytest

from aegisflow.rag.types import Chunk
from aegisflow.rag.vector_index import PgVectorIndex


DSN = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")


def _pg_up() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_up(), reason="Postgres not reachable")


def test_pgvector_counts_source_filter_and_query_latency():
    index = PgVectorIndex(DSN)
    index.initialize()
    strategy = "test-" + uuid.uuid4().hex
    chunks = [
        Chunk(
            "fastapi/tutorial/background-tasks#0",
            "fastapi/tutorial/background-tasks",
            "How to add a background task",
            {
                "source": "fastapi",
                "heading_path": ["Tutorial", "Background Tasks"],
                "chunk_strategy": strategy,
                "token_count": 8,
            },
        ),
        Chunk(
            "langchain/guide#0",
            "langchain/guide",
            "A guide to another topic",
            {
                "source": "langchain",
                "heading_path": ["Guide"],
                "chunk_strategy": strategy,
                "token_count": 6,
            },
        ),
    ]
    fastapi_vector = [1.0] + [0.0] * 383
    other_vector = [0.0, 1.0] + [0.0] * 382
    query_vector = fastapi_vector

    try:
        vectors = [fastapi_vector, other_vector]
        doc_ids = [chunk.doc_id for chunk in chunks]
        index.replace_strategy(
            chunks,
            vectors,
            strategy=strategy,
            doc_ids=doc_ids,
            batch_size=1,
        )
        assert index.count(strategy) == len(chunks)
        filtered = index.search(query_vector, strategy=strategy, source="fastapi", k=5)
        assert filtered and filtered[0][0] == chunks[0].id
        assert len(filtered) == 1
        assert index.explain_search(query_vector, strategy=strategy, source="fastapi")

        durations = []
        for _ in range(100):
            started = time.perf_counter()
            index.search(query_vector, strategy=strategy, k=3)
            durations.append(time.perf_counter() - started)
        p95 = sorted(durations)[94]
        print(f"pgvector query latency p95 (100 queries): {p95 * 1000:.1f} ms")
        assert p95 < 0.1, f"query p95 was {p95 * 1000:.1f} ms"
        index.replace_strategy(
            chunks[:1],
            vectors[:1],
            strategy=strategy,
            doc_ids=doc_ids,
        )
        assert index.count(strategy) == 1
    finally:
        with psycopg.connect(DSN) as connection:
            connection.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
