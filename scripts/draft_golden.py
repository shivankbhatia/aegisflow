"""Draft a source-balanced RAG golden set from indexed structure-aware chunks."""

from __future__ import annotations

import asyncio
import os
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.llm_gateway import from_env
from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import GoldenQuestion, load_questions, save_questions
from aegisflow.rag.llm_eval import generate_questions
from aegisflow.rag.types import Chunk
from aegisflow.rag.vector_index import PgVectorIndex


STRATEGY = "structure-aware"
SEED = 20261024


def load_indexed_chunks(index: PgVectorIndex) -> list[Chunk]:
    with index._connect() as connection:
        rows = connection.execute(
            "SELECT id, doc_id, source, heading_path, text, token_count "
            "FROM chunks WHERE strategy = %s ORDER BY source, doc_id, id",
            (STRATEGY,),
        ).fetchall()
    return [
        Chunk(
            id=row[0],
            doc_id=row[1],
            text=row[4],
            metadata={
                "source": row[2],
                "heading_path": row[3],
                "token_count": row[5],
                "chunk_strategy": STRATEGY,
            },
        )
        for row in rows
    ]


def choose_single_chunks(chunks: list[Chunk], count: int = 80) -> list[Chunk]:
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.metadata["token_count"] >= 100:
            by_source[chunk.metadata["source"]].append(chunk)
    sources = sorted(by_source)
    if len(sources) < 3 or any(len(by_source[source]) < count // len(sources) for source in sources):
        raise RuntimeError(f"not enough eligible chunks for balanced sampling: {dict((s, len(v)) for s, v in by_source.items())}")
    rng = random.Random(SEED)
    per_source = count // len(sources)
    remainder = count % len(sources)
    selected = []
    for i, source in enumerate(sources):
        candidates = by_source[source]
        rng.shuffle(candidates)
        selected.extend(candidates[: per_source + (i < remainder)])
    return selected


def choose_multihop_pairs(
    chunks: list[Chunk], index: PgVectorIndex, embedder: BGEEmbedder, count: int = 30
) -> list[tuple[Chunk, Chunk]]:
    by_id = {chunk.id: chunk for chunk in chunks}
    pairs: list[tuple[Chunk, Chunk]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if len(pairs) >= count:
            break
        vector = embedder.embed_query(chunk.text)
        neighbours = index.search(vector, strategy=STRATEGY, k=30)
        candidates = [by_id[cid] for cid, _ in neighbours if cid in by_id]
        candidates.sort(key=lambda other: (other.metadata["source"] == chunk.metadata["source"], other.doc_id == chunk.doc_id))
        for other in candidates:
            if other.doc_id == chunk.doc_id:
                continue
            key = tuple(sorted((chunk.id, other.id)))
            if key not in seen:
                seen.add(key)
                pairs.append((chunk, other))
                break
    if len(pairs) != count:
        raise RuntimeError(f"found only {len(pairs)} cross-document pairs, expected {count}")
    return pairs


async def main() -> None:
    load_dotenv()
    dsn = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")
    index = PgVectorIndex(dsn)
    count = await asyncio.to_thread(index.count, STRATEGY)
    if not count:
        raise RuntimeError("structure-aware index is empty; run scripts/index_rag_corpus.py first")
    chunks = await asyncio.to_thread(load_indexed_chunks, index)
    print(f"Loaded {len(chunks)} structure-aware chunks from PostgreSQL")
    singles = choose_single_chunks(chunks)
    embedder = BGEEmbedder()
    pairs = await asyncio.to_thread(choose_multihop_pairs, chunks, index, embedder)
    client = from_env(max_retries=8, backoff_base=5.0, backoff_cap=30.0)
    try:
        checkpoint = Path("eval/golden.draft.jsonl")
        drafts: list[GoldenQuestion] = load_questions(checkpoint) if checkpoint.exists() else []
        existing_ids = {question.id for question in drafts}
        # Single in-flight requests stay within the configured provider's TPM cap.
        # Checkpoint after each small batch so a transient provider failure can resume.
        for start in range(0, len(singles), 8):
            prefix = f"s{start:02d}_"
            if any(question_id.startswith(prefix) for question_id in existing_ids):
                continue
            batch = await generate_questions(
                client,
                [[chunk] for chunk in singles[start : start + 8]],
                kind="single",
                id_prefix=prefix,
                concurrency=1,
            )
            drafts.extend(batch)
            existing_ids.update(question.id for question in batch)
            save_questions(drafts, checkpoint)
        for start in range(0, len(pairs), 8):
            prefix = f"m{start:02d}_"
            if any(question_id.startswith(prefix) for question_id in existing_ids):
                continue
            batch = await generate_questions(
                client,
                [list(pair) for pair in pairs[start : start + 8]],
                kind="multi_hop",
                id_prefix=prefix,
                concurrency=1,
            )
            drafts.extend(batch)
            existing_ids.update(question.id for question in batch)
            save_questions(drafts, checkpoint)
    finally:
        await client.aclose()
    counts = {kind: sum(question.kind == kind for question in drafts) for kind in ("single", "multi_hop")}
    if counts != {"single": 80, "multi_hop": 30}:
        raise RuntimeError(f"incomplete question draft pool: {counts}; checkpoint retained at {checkpoint}")
    path = Path("eval/golden.jsonl")
    save_questions(drafts, path)
    checkpoint.unlink(missing_ok=True)
    print(f"Saved {len(drafts)} unverified drafts to {path}")
    for source in sorted({c.metadata["source"] for c in singles}):
        print(f"single {source}: {sum(c.metadata['source'] == source for c in singles)} sampled")
    print(f"multi-hop cross-document pairs: {len(pairs)}")
    print("Next: run scripts/review_golden.py; add 15 questions written from your own documentation doubts.")


if __name__ == "__main__":
    asyncio.run(main())
