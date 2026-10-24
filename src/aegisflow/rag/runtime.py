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
from .vector_index import PgVectorIndex
from .chunkers import structure_aware_chunks
from .clean import clean_markdown
from .corpus import iter_documents


@dataclass(frozen=True)
class RAGRuntime:
    index: PgVectorIndex
    embedder: BGEEmbedder
    reranker: CrossEncoderReranker
    retrieve: Callable[[str, int], Awaitable[list[str]]]


async def load_rag_runtime(dsn: str) -> RAGRuntime:
    """Load the embedder and cross encoder once, off the event loop."""
    # Serialize both models on the shared Metal command stream.
    inference_lock = threading.Lock()
    device = os.getenv("AEGIS_RAG_DEVICE") or None
    candidate_k = int(os.getenv("AEGIS_RAG_CANDIDATE_K", "30"))
    if not 1 <= candidate_k <= 50:
        raise ValueError("AEGIS_RAG_CANDIDATE_K must be between 1 and 50")
    embedder = await asyncio.to_thread(BGEEmbedder, device=device, inference_lock=inference_lock)
    reranker = await asyncio.to_thread(CrossEncoderReranker, device=device, inference_lock=inference_lock)
    index = PgVectorIndex(dsn)
    retrieve = make_reranked_hybrid_retriever(
        index,
        embedder.embed_query,
        reranker,
        strategy="structure-aware",
        candidate_k=candidate_k,
        dense_k=50,
        sparse_k=50,
    )
    return RAGRuntime(index=index, embedder=embedder, reranker=reranker, retrieve=retrieve)


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
