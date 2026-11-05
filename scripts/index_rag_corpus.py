"""Embed, index, and smoke-test the pinned documentation corpus."""

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path

from aegisflow.rag.chunkers import (
    fixed_size_chunks,
    semantic_chunks,
    structure_aware_chunks,
)
from aegisflow.rag.clean import clean_markdown
from aegisflow.rag.corpus import iter_documents
from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.vector_index import PgVectorIndex, make_retriever


SOURCES = {
    "langchain": Path("data/raw/langchain-docs/src/oss"),
    "fastapi": Path("data/raw/fastapi/docs/en/docs"),
    "psycopg": Path("data/raw/psycopg/docs"),
}
STRATEGIES = ("fixed-size", "structure-aware", "semantic")
EMBED_BATCH_SIZE = 256


async def main() -> None:
    dsn = os.getenv(
        "DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow"
    )
    index = PgVectorIndex(dsn)
    await asyncio.to_thread(index.initialize)
    embedder = BGEEmbedder()

    chunks_by_strategy = {name: [] for name in STRATEGIES}
    document_ids: list[str] = []
    for source, docs_dir in SOURCES.items():
        for doc_id, _source, relative_path, raw_text in iter_documents(docs_dir, source):
            document_ids.append(doc_id)
            title, sections = clean_markdown(raw_text, relative_path)
            chunks_by_strategy["fixed-size"].extend(
                fixed_size_chunks(
                    sections, doc_id=doc_id, source=source, title=title
                )
            )
            chunks_by_strategy["structure-aware"].extend(
                structure_aware_chunks(
                    sections, doc_id=doc_id, source=source, title=title
                )
            )
            chunks_by_strategy["semantic"].extend(
                await asyncio.to_thread(
                    semantic_chunks,
                    sections,
                    doc_id=doc_id,
                    source=source,
                    title=title,
                    embed_documents=embedder.embed_documents,
                )
            )

    for strategy, chunks in chunks_by_strategy.items():
        vectors = []
        for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[offset : offset + EMBED_BATCH_SIZE]
            vectors.extend(await embedder.aembed_documents([chunk.text for chunk in batch]))
        await asyncio.to_thread(
            index.replace_strategy,
            chunks,
            vectors,
            strategy=strategy,
            doc_ids=document_ids,
            batch_size=EMBED_BATCH_SIZE,
        )
        row_count = index.count(strategy)
        if row_count != len(chunks):
            raise RuntimeError(
                f"{strategy} indexed {row_count} rows for {len(chunks)} chunks"
            )
        print(f"{strategy}: chunks={len(chunks)}, rows={row_count}")

    query = "how do I add a background task"
    query_vector = await embedder.aembed_query(query)
    sanity = index.search(query_vector, strategy="fixed-size", k=3)
    print(f"sanity top 3 for {query!r}: {sanity}")
    if not any(chunk_id.startswith("fastapi/tutorial/background-tasks#") for chunk_id, _ in sanity):
        raise RuntimeError("FastAPI background-tasks page did not appear in the top 3")
    fastapi_only = index.search(
        query_vector, strategy="fixed-size", source="fastapi", k=10
    )
    if any(not chunk_id.startswith("fastapi/") for chunk_id, _ in fastapi_only):
        raise RuntimeError("source=fastapi filter returned a non-FastAPI chunk")

    retriever = make_retriever(index, embedder.embed_query, strategy="fixed-size")
    durations = []
    for _ in range(100):
        started = time.perf_counter()
        await retriever(query, 3)
        durations.append(time.perf_counter() - started)
    p95 = sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
    print(f"100-query latency p95: {p95 * 1000:.1f} ms")
    if p95 >= 0.1:
        raise RuntimeError(f"100-query p95 exceeded 100 ms: {p95 * 1000:.1f} ms")
    plan = index.explain_search(query_vector, strategy="fixed-size")
    relevant_plan = [line.strip() for line in plan if "Index Scan" in line or "Filter:" in line]
    print("EXPLAIN:", "; ".join(relevant_plan))
    if not any("chunks_embedding_hnsw_idx" in line for line in plan):
        raise RuntimeError("EXPLAIN did not select the HNSW index")


if __name__ == "__main__":
    asyncio.run(main())
