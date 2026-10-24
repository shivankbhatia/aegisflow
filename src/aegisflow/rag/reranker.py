"""Cross-encoder reranking for hybrid RAG retrieval."""

from __future__ import annotations

import asyncio
import platform
import threading
from collections.abc import Callable, Sequence
from typing import Awaitable

from .vector_index import PgVectorIndex, reciprocal_rank_fusion


RERANKER_MODEL = "BAAI/bge-reranker-base"
MAX_RERANK_TOKENS = 512
RERANK_BATCH_SIZE = 16


class CrossEncoderReranker:
    """Load one BGE cross encoder and score query/passage pairs in batches."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = RERANK_BATCH_SIZE,
        max_length: int = MAX_RERANK_TOKENS,
        model=None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if device is None:
            device = self._preferred_device()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device=device, max_length=max_length)
        self._model = model

    @staticmethod
    def _preferred_device() -> str:
        if platform.system() == "Darwin":
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "mps"
            except (ImportError, AttributeError):
                pass
        return "cpu"

    def score(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """Return candidate IDs with cross-encoder scores, highest first."""
        if not candidates:
            return []
        pairs = [(query, text) for _chunk_id, text in candidates]
        # CrossEncoder truncates tokenized pairs at max_length. A lock keeps
        # concurrent requests from overlapping inference on a shared MPS model.
        with self._lock:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        return sorted(ranked, key=lambda item: -item[1])

    async def ascore(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Score pairs off the event loop."""
        return await asyncio.to_thread(self.score, query, candidates)


def make_reranked_hybrid_retriever(
    index: PgVectorIndex,
    embed_query: Callable[[str], Sequence[float]],
    reranker: CrossEncoderReranker,
    *,
    strategy: str,
    source: str | None = None,
    candidate_k: int = 50,
    dense_k: int = 50,
    sparse_k: int = 50,
    k_rrf: int = 60,
    weights: Sequence[float] = (1.0, 0.5),
) -> Callable[[str, int], Awaitable[list[str]]]:
    """Create a ``(query, k) -> ids`` adapter: hybrid candidates, then rerank."""
    if min(candidate_k, dense_k, sparse_k) < 1:
        raise ValueError("candidate and retrieval limits must be positive")
    if len(weights) != 2:
        raise ValueError("hybrid weights must contain dense and sparse weights")

    async def retrieve(query: str, k: int) -> list[str]:
        if k < 1:
            raise ValueError("k must be positive")

        def fetch_candidates() -> list[tuple[str, str]]:
            vector = embed_query(query)
            dense = index.search(vector, strategy=strategy, k=dense_k, source=source)
            sparse = index.sparse_search(query, strategy=strategy, k=sparse_k, source=source)
            fused = reciprocal_rank_fusion(
                [chunk_id for chunk_id, _ in dense],
                [chunk_id for chunk_id, _ in sparse],
                k_rrf=k_rrf,
                weights=weights,
                limit=candidate_k,
            )
            ids = [chunk_id for chunk_id, _ in fused]
            texts = index.fetch_texts(ids, strategy=strategy)
            return [(chunk_id, texts[chunk_id]) for chunk_id in ids if chunk_id in texts]

        candidates = await asyncio.to_thread(fetch_candidates)
        ranked = await reranker.ascore(query, candidates)
        return [chunk_id for chunk_id, _score in ranked[:k]]

    return retrieve
