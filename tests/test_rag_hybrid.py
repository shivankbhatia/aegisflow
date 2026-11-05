import os
import uuid

import psycopg
import pytest

from aegisflow.rag.types import Chunk
from aegisflow.rag.vector_index import PgVectorIndex, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_orders_overlap_and_keeps_single_list_items():
    fused = reciprocal_rank_fusion(
        ["a", "b", "dense-only"],
        ["b", "a", "sparse-only"],
        k_rrf=60,
    )
    scores = dict(fused)
    assert fused[:2] == [("a", scores["a"]), ("b", scores["b"])]  # stable dense-order tie-break
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["dense-only"] == pytest.approx(1 / 63)
    assert scores["sparse-only"] == pytest.approx(1 / 63)
    assert len(fused) == 4
    weighted = reciprocal_rank_fusion(["dense"], ["sparse"], weights=[1.0, 0.5])
    assert dict(weighted) == pytest.approx({"dense": 1 / 61, "sparse": 0.5 / 61})


@pytest.mark.parametrize("arguments", [
    {"k_rrf": 0},
    {"weights": [1.0]},
    {"weights": [-1.0, 1.0]},
])
def test_reciprocal_rank_fusion_rejects_invalid_configuration(arguments):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(["a"], ["b"], **arguments)


DSN = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")


def _pg_up() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_up(), reason="Postgres not reachable")
def test_sparse_search_finds_fastapi_background_tasks_for_exact_identifiers():
    index = PgVectorIndex(DSN)
    index.initialize()
    strategy = "sparse-test-" + uuid.uuid4().hex
    chunk = Chunk(
        "fastapi/tutorial/background-tasks#test",
        "fastapi/tutorial/background-tasks",
        "FastAPI BackgroundTasks lets you add_task to run work after a response.",
        {
            "source": "fastapi",
            "heading_path": ["Tutorial", "Background Tasks"],
            "chunk_strategy": strategy,
            "token_count": 12,
        },
    )
    try:
        index.replace_strategy(
            [chunk], [[0.0] * 384], strategy=strategy,
            doc_ids=[chunk.doc_id],
        )
        rows = index.sparse_search(
            "BackgroundTasks add_task",
            strategy=strategy,
            source="fastapi",
            k=10,
        )
        assert rows
        assert rows[0][0] == chunk.id
    finally:
        with psycopg.connect(DSN) as connection:
            connection.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
