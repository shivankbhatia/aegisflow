from contextlib import asynccontextmanager

import pytest

from aegisflow.api import main as api_main
from aegisflow.rag.runtime import (
    DEFAULT_RAG_MODE,
    DEFAULT_RAG_STRATEGY,
    make_mode_retriever,
    rag_config_from_env,
)


class FakeIndex:
    def search(self, _embedding, *, strategy, k, source=None):
        return [(f"{strategy}/page-{i}#0", 1.0 - i / 10) for i in range(min(k, 3))]

    def sparse_search(self, _query, *, strategy, k, source=None):
        return [(f"{strategy}/page-{i}#0", 1.0 - i / 10) for i in range(min(k, 3))]

    def fetch_texts(self, ids, *, strategy):
        return {chunk_id: f"text {chunk_id}" for chunk_id in ids}


class FakeReranker:
    async def ascore(self, _query, candidates):
        return [(chunk_id, float(len(candidates) - i)) for i, (chunk_id, _) in enumerate(candidates)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dense", "hybrid", "hybrid_rerank"])
async def test_each_mode_retrieves_with_fakes(mode):
    retrieve = make_mode_retriever(
        FakeIndex(), lambda _query: [0.0] * 384, mode=mode,
        strategy="structure-aware", reranker=FakeReranker(), candidate_k=5,
    )
    ids = await retrieve("query", 2, candidate_limit=2)
    assert len(ids) == 2
    assert all("structure-aware/page-" in chunk_id for chunk_id in ids)


def test_default_served_configuration_matches_selection(monkeypatch):
    monkeypatch.delenv("AEGIS_RAG_MODE", raising=False)
    monkeypatch.delenv("AEGIS_RAG_STRATEGY", raising=False)
    assert rag_config_from_env() == (DEFAULT_RAG_MODE, DEFAULT_RAG_STRATEGY)
    assert rag_config_from_env() == ("dense", "structure-aware")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "value"),
    [("AEGIS_RAG_MODE", "not-a-mode"), ("AEGIS_RAG_STRATEGY", "not-a-strategy")],
)
async def test_invalid_config_fails_startup_before_database(monkeypatch, setting, value):
    monkeypatch.setenv("AEGIS_RAG_ENABLED", "1")
    monkeypatch.setenv(setting, value)
    called = False

    @asynccontextmanager
    async def database_must_not_open(_dsn):
        nonlocal called
        called = True
        raise AssertionError("database opened before RAG config validation")
        yield  # pragma: no cover

    monkeypatch.setattr(api_main, "postgres_checkpointer", database_must_not_open)
    app = api_main.create_app()
    with pytest.raises(ValueError, match=setting):
        async with app.router.lifespan_context(app):
            pass
    assert not called
