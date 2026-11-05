"""Compare hybrid retrieval with BGE cross-encoder reranking."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import evaluate, load_questions, save_report
from aegisflow.rag.reranker import CrossEncoderReranker, make_reranked_hybrid_retriever
from aegisflow.rag.vector_index import PgVectorIndex, make_hybrid_retriever


async def run(candidate_k: int, index, embedder, reranker, questions):
    reranked = make_reranked_hybrid_retriever(
        index,
        embedder.embed_query,
        reranker,
        strategy="structure-aware",
        candidate_k=candidate_k,
        dense_k=50,
        sparse_k=50,
    )
    return await evaluate(reranked, questions, ks=(1, 5, 10), concurrency=1)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("eval/golden.jsonl"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=50)
    args = parser.parse_args()
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be positive")
    load_dotenv()
    questions = load_questions(args.questions, only_verified=not args.allow_unverified)
    if not questions:
        raise SystemExit("No verified questions found")

    dsn = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")
    index = PgVectorIndex(dsn)
    embedder = BGEEmbedder()
    reranker = CrossEncoderReranker()
    print(f"reranker device: {reranker.device}; candidates: {args.candidate_count}")
    hybrid = make_hybrid_retriever(
        index,
        embedder.embed_query,
        strategy="structure-aware",
        dense_k=50,
        sparse_k=50,
    )
    hybrid_report = await evaluate(hybrid, questions, ks=(1, 5, 10), concurrency=1)
    reranked_report = await run(args.candidate_count, index, embedder, reranker, questions)

    candidate_k = args.candidate_count
    added_p95 = reranked_report["overall"]["latency_p95_s"] - hybrid_report["overall"]["latency_p95_s"]
    initial_reranked_path = None
    if candidate_k == 50 and added_p95 >= 0.300:
        print("50-candidate reranking added at least 300 ms p95; retrying with 30 candidates")
        initial_reranked_path = save_report(reranked_report, "hybrid-reranked-50-candidates")
        print(
            f"50-candidate results saved: {initial_reranked_path}; "
            f"MRR={reranked_report['overall']['mrr']:.3f}; "
            f"added p95={added_p95 * 1000:.1f} ms"
        )
        candidate_k = 30
        reranked_report = await run(candidate_k, index, embedder, reranker, questions)
        added_p95 = reranked_report["overall"]["latency_p95_s"] - hybrid_report["overall"]["latency_p95_s"]

    hybrid_path = save_report(hybrid_report, "hybrid-before-reranker")
    reranked_path = save_report(reranked_report, f"hybrid-reranked-{candidate_k}-candidates")
    mrr_lift = reranked_report["overall"]["mrr"] - hybrid_report["overall"]["mrr"]
    latency_ms = reranked_report["overall"]["latency_p95_s"] * 1000
    print(f"hybrid report: {hybrid_path}")
    if initial_reranked_path:
        print(f"50-candidate report: {initial_reranked_path}")
    print(f"reranked report: {reranked_path}")
    print(
        f"MRR: hybrid={hybrid_report['overall']['mrr']:.3f}, "
        f"reranked={reranked_report['overall']['mrr']:.3f}, lift={mrr_lift:+.3f}"
    )
    print(
        f"reranked recall@10={reranked_report['overall']['recall@10']:.3f}; "
        f"p95={latency_ms:.1f} ms; added p95={added_p95 * 1000:.1f} ms; candidates={candidate_k}"
    )

    changed = []
    before = {row["id"]: row for row in hybrid_report["per_question"]}
    after = {row["id"]: row for row in reranked_report["per_question"]}
    for question in questions:
        old = before[question.id]["retrieved"]
        new = after[question.id]["retrieved"]
        if old != new:
            changed.append((question, old, new))
    spot_path = Path("eval/reranker_spot_checks.md")
    lines = [
        "# Reranker spot checks",
        "",
        f"Candidate count: {candidate_k}; reranked p95: {latency_ms:.1f} ms; "
        f"added p95: {added_p95 * 1000:.1f} ms.",
        "For each changed query, compare the gold passages with the before/after ranking.",
        "",
    ]
    for question, old, new in changed[:5]:
        lines.extend(
            [
                f"## {question.id}: {question.question}",
                f"- Gold: {', '.join(question.gold_chunk_ids)}",
                f"- Hybrid top 5: {', '.join(old[:5])}",
                f"- Reranked top 5: {', '.join(new[:5])}",
                "- Assessment: ",
                "",
            ]
        )
    if not changed:
        lines.append("No rankings changed; no spot checks are available.")
    elif len(changed) < 5:
        lines.append(f"Only {len(changed)} queries changed ranking.")
    spot_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"spot-check worksheet: {spot_path} ({min(5, len(changed))} changed queries)")

    if added_p95 >= 0.300:
        print("WARNING: added p95 remains above the 300 ms target at the reduced candidate count")


if __name__ == "__main__":
    asyncio.run(main())
