"""Long-lived retrieval components suitable for loading during app lifespan."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from .embeddings import BGEEmbedder
from .reranker import CrossEncoderReranker, make_reranked_hybrid_retriever
from .vector_index import PgVectorIndex


@dataclass(frozen=True)
class RAGRuntime:
    index: PgVectorIndex
    embedder: BGEEmbedder
    reranker: CrossEncoderReranker
    retrieve: Callable[[str, int], Awaitable[list[str]]]


async def load_rag_runtime(dsn: str) -> RAGRuntime:
    """Load the embedder and cross encoder once, off the event loop."""
    embedder = await asyncio.to_thread(BGEEmbedder)
    reranker = await asyncio.to_thread(CrossEncoderReranker)
    index = PgVectorIndex(dsn)
    retrieve = make_reranked_hybrid_retriever(
        index,
        embedder.embed_query,
        reranker,
        strategy="structure-aware",
        candidate_k=50,
        dense_k=50,
        sparse_k=50,
    )
    return RAGRuntime(index=index, embedder=embedder, reranker=reranker, retrieve=retrieve)
