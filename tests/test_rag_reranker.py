import asyncio

import numpy as np
import pytest

from aegisflow.rag.reranker import CrossEncoderReranker, make_reranked_hybrid_retriever


class FakeCrossEncoder:
    def __init__(self, scores=None):
        self.scores = scores or []
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        if self.scores:
            return np.asarray(self.scores[: len(pairs)])
        return np.asarray([float(i) for i in range(len(pairs))])


def test_cross_encoder_scores_in_configured_batches_and_sorts_descending():
    model = FakeCrossEncoder([0.1, 0.9, 0.4])
    reranker = CrossEncoderReranker(model=model, batch_size=16, max_length=512)

    ranked = reranker.score("query", [("a", "A"), ("b", "B"), ("c", "C")])

    assert ranked == [("b", 0.9), ("c", 0.4), ("a", 0.1)]
    pairs, kwargs = model.calls[0]
    assert pairs == [("query", "A"), ("query", "B"), ("query", "C")]
    assert kwargs["batch_size"] == 16
    assert kwargs["show_progress_bar"] is False
    assert reranker.max_length == 512


def test_cross_encoder_handles_empty_candidates_and_invalid_configuration():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    assert reranker.score("query", []) == []
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker(model=FakeCrossEncoder(), batch_size=0)
    with pytest.raises(ValueError, match="max_length"):
        CrossEncoderReranker(model=FakeCrossEncoder(), max_length=0)


class FakeIndex:
    def search(self, vector, *, strategy, k, source):
        assert vector == [1.0]
        assert strategy == "structure-aware"
        assert k == 50
        assert source is None
        return [("dense-a", 0.9), ("shared", 0.8)]

    def sparse_search(self, query, *, strategy, k, source):
        assert query == "query"
        return [("shared", 1.0), ("sparse-b", 0.5)]

    def fetch_texts(self, ids, *, strategy):
        assert strategy == "structure-aware"
        return {chunk_id: f"text {chunk_id}" for chunk_id in ids}


def test_reranked_hybrid_retriever_returns_reranker_order_and_respects_k():
    model = FakeCrossEncoder([0.1, 0.9, 0.4])
    reranker = CrossEncoderReranker(model=model)
    retrieve = make_reranked_hybrid_retriever(
        FakeIndex(), lambda _query: [1.0], reranker,
        strategy="structure-aware", candidate_k=3,
    )

    assert asyncio.run(retrieve("query", 2)) == ["dense-a", "sparse-b"]


def test_reranked_hybrid_retriever_can_limit_candidates_for_low_latency():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)
    retrieve = make_reranked_hybrid_retriever(
        FakeIndex(), lambda _query: [1.0], reranker,
        strategy="structure-aware", candidate_k=3,
    )
    assert asyncio.run(retrieve("query", 2, candidate_limit=2)) == ["dense-a", "shared"]
    pairs, _kwargs = model.calls[0]
    assert pairs == [("query", "text shared"), ("query", "text dense-a")]


def test_reranked_hybrid_retriever_rejects_bad_candidate_limits():
    with pytest.raises(ValueError, match="limits"):
        make_reranked_hybrid_retriever(
            FakeIndex(), lambda _query: [1.0], CrossEncoderReranker(model=FakeCrossEncoder()),
            strategy="structure-aware", candidate_k=0,
        )
