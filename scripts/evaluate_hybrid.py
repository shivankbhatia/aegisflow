"""Compare dense-only and RRF hybrid retrieval on a golden-question file."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import evaluate, load_questions, save_report
from aegisflow.rag.vector_index import PgVectorIndex, make_hybrid_retriever, make_retriever


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("eval/golden.jsonl"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--sparse-weight", type=float, default=0.5)
    args = parser.parse_args()
    load_dotenv()
    questions = load_questions(args.questions, only_verified=not args.allow_unverified)
    if not questions:
        raise SystemExit("No verified questions found. Use --allow-unverified for a clearly provisional draft comparison.")
    if any(not question.verified for question in questions):
        print(f"WARNING: provisional comparison over {len(questions)} unverified draft questions")

    dsn = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")
    index = PgVectorIndex(dsn)
    embedder = BGEEmbedder()
    dense = make_retriever(index, embedder.embed_query, strategy="structure-aware")
    hybrid = make_hybrid_retriever(
        index,
        embedder.embed_query,
        strategy="structure-aware",
        weights=(1.0, args.sparse_weight),
    )
    # Keep local sentence-transformer inference serial on Apple Silicon; some
    # MPS/PyTorch builds crash when concurrent encode calls overlap.
    dense_report = await evaluate(dense, questions, ks=(10,), concurrency=1)
    hybrid_report = await evaluate(hybrid, questions, ks=(10,), concurrency=1)
    dense_path = save_report(dense_report, "dense-structure-aware")
    hybrid_path = save_report(
        hybrid_report, f"hybrid-structure-aware-sparse-{args.sparse_weight:g}"
    )
    print(f"dense report: {dense_path}")
    print(f"hybrid report: {hybrid_path}")
    dense_recall = dense_report["overall"]["recall@10"]
    hybrid_recall = hybrid_report["overall"]["recall@10"]
    lift = hybrid_recall - dense_recall
    print(f"overall recall@10: dense={dense_recall:.3f}, hybrid={hybrid_recall:.3f}, lift={lift:+.3f}")
    kinds = sorted(set(dense_report["by_kind"]) | set(hybrid_report["by_kind"]))
    for kind in kinds:
        d = dense_report["by_kind"].get(kind, {}).get("recall@10", float("nan"))
        h = hybrid_report["by_kind"].get(kind, {}).get("recall@10", float("nan"))
        print(f"{kind} recall@10: dense={d:.3f}, hybrid={h:.3f}, lift={h - d:+.3f}")
    print(f"hybrid end-to-end p95: {hybrid_report['overall']['latency_p95_s'] * 1000:.1f} ms")
    if hybrid_report["overall"]["latency_p95_s"] >= 0.150:
        raise SystemExit("hybrid p95 exceeded 150 ms")


if __name__ == "__main__":
    asyncio.run(main())
