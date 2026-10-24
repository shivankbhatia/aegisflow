"""Long-lived retrieval components suitable for loading during app lifespan."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Awaitable, Callable

from .embeddings import BGEEmbedder
from .reranker import CrossEncoderReranker, make_reranked_hybrid_retriever
from .vector_index import (
    PgVectorIndex,
    make_hybrid_retriever,
    make_retriever,
)
from .chunkers import structure_aware_chunks
from .clean import clean_markdown
from .corpus import iter_documents


@dataclass(frozen=True)
class RAGRuntime:
    index: PgVectorIndex
    embedder: BGEEmbedder
    reranker: CrossEncoderReranker | None
    retrieve: Callable[[str, int], Awaitable[list[str]]]
    mode: str = "dense"
    strategy: str = "structure-aware"


DEFAULT_RAG_MODE = "dense"
DEFAULT_RAG_STRATEGY = "structure-aware"
RAG_MODES = {"dense", "hybrid", "hybrid_rerank"}
RAG_STRATEGIES = {"fixed-size", "structure-aware", "semantic"}


def rag_config_from_env() -> tuple[str, str]:
    mode = os.getenv("AEGIS_RAG_MODE", DEFAULT_RAG_MODE)
    strategy = os.getenv("AEGIS_RAG_STRATEGY", DEFAULT_RAG_STRATEGY)
    if mode not in RAG_MODES:
        raise ValueError(f"AEGIS_RAG_MODE must be one of {sorted(RAG_MODES)}")
    if strategy not in RAG_STRATEGIES:
        raise ValueError(f"AEGIS_RAG_STRATEGY must be one of {sorted(RAG_STRATEGIES)}")
    return mode, strategy


def make_mode_retriever(index, embed_query, *, mode, strategy, reranker=None, candidate_k=30):
    """Build one served retrieval mode for a configured chunking strategy."""
    if mode not in RAG_MODES:
        raise ValueError(f"invalid RAG mode: {mode}")
    if strategy not in RAG_STRATEGIES:
        raise ValueError(f"invalid RAG strategy: {strategy}")
    if mode == "dense":
        base = make_retriever(index, embed_query, strategy=strategy)
    elif mode == "hybrid":
        base = make_hybrid_retriever(index, embed_query, strategy=strategy)
    else:
        if reranker is None:
            raise ValueError("hybrid_rerank mode requires a loaded reranker")
        base = make_reranked_hybrid_retriever(
            index, embed_query, reranker, strategy=strategy, candidate_k=candidate_k,
        )

    async def retrieve(query, k, *, timing_callback=None, candidate_limit=None):
        if mode == "hybrid_rerank":
            return await base(
                query, k, timing_callback=timing_callback,
                candidate_limit=candidate_limit,
            )
        return await base(query, k)

    return retrieve


async def load_rag_runtime(dsn: str) -> RAGRuntime:
    """Load only the selected retrieval models once, off the event loop."""
    mode, strategy = rag_config_from_env()
    # Serialize both models on the shared Metal command stream.
    inference_lock = threading.Lock()
    device = os.getenv("AEGIS_RAG_DEVICE") or None
    candidate_k = int(os.getenv("AEGIS_RAG_CANDIDATE_K", "30"))
    if mode == "hybrid_rerank" and not 1 <= candidate_k <= 50:
        raise ValueError("AEGIS_RAG_CANDIDATE_K must be between 1 and 50")
    embedder = await asyncio.to_thread(BGEEmbedder, device=device, inference_lock=inference_lock)
    reranker = None
    if mode == "hybrid_rerank":
        reranker = await asyncio.to_thread(
            CrossEncoderReranker, device=device, inference_lock=inference_lock,
        )
    index = PgVectorIndex(dsn)
    retrieve = make_mode_retriever(
        index, embedder.embed_query, mode=mode, strategy=strategy,
        reranker=reranker, candidate_k=candidate_k,
    )
    return RAGRuntime(
        index=index, embedder=embedder, reranker=reranker, retrieve=retrieve,
        mode=mode, strategy=strategy,
    )


def ingest_local_corpus(runtime: RAGRuntime, root: str | Path = "data/raw") -> dict[str, int]:
    """Build and transactionally replace the structure-aware local corpus index."""
    docs_root = Path(root)
    chunks = []
    doc_ids: list[str] = []
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    for source_dir in sorted(path for path in docs_root.iterdir() if path.is_dir()):
        docs_path = source_dir / "docs"
        if not docs_path.is_dir():
            # LangChain docs repository stores its content below docs.
            docs_path = source_dir
        for doc_id, source, relative_path, raw in iter_documents(docs_path, source_dir.name):
            title, sections = clean_markdown(raw, relative_path)
            chunks.extend(structure_aware_chunks(
                sections, doc_id=doc_id, source=source, title=title,
                token_counter=lambda text: len(encoder.encode(text, disallowed_special=())),
            ))
            doc_ids.append(doc_id)
    vectors = runtime.embedder.embed_documents([chunk.text for chunk in chunks])
    runtime.index.replace_strategy(chunks, vectors, strategy="structure-aware", doc_ids=doc_ids)
    return {"documents": len(doc_ids), "chunks": len(chunks)}
