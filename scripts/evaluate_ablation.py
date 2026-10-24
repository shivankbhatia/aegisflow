"""Run the final dense/chunker, hybrid, and reranker retrieval ablation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.corpus import iter_documents
from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import evaluate, format_comparison, load_questions, save_report
from aegisflow.rag.reranker import CrossEncoderReranker, make_reranked_hybrid_retriever
from aegisflow.rag.vector_index import PgVectorIndex, make_hybrid_retriever, make_retriever


SOURCES = {
    "langchain": Path("data/raw/langchain-docs/src/oss"),
    "fastapi": Path("data/raw/fastapi/docs/en/docs"),
    "psycopg": Path("data/raw/psycopg/docs"),
}
CHUNKERS = ("fixed-size", "structure-aware", "semantic")


def count_pages() -> tuple[int, dict[str, int]]:
    by_source = {
        source: sum(1 for _ in iter_documents(path, source))
        for source, path in SOURCES.items()
    }
    return sum(by_source.values()), by_source


async def main() -> None:
    load_dotenv()
    questions = load_questions("eval/golden.jsonl", only_verified=True)
    if not questions:
        raise SystemExit("No verified questions found")

    dsn = os.getenv("DATABASE_URL", "postgresql://aegis:password@localhost:5432/aegisflow")
    index = PgVectorIndex(dsn)
    embedder = BGEEmbedder()
    chunker_reports: dict[str, dict] = {}
    report_paths: dict[str, str] = {}

    for strategy in CHUNKERS:
        count = index.count(strategy)
        if count == 0:
            raise RuntimeError(f"No indexed chunks for strategy {strategy!r}")
        retriever = make_retriever(index, embedder.embed_query, strategy=strategy)
        report = await evaluate(retriever, questions, ks=(1, 5, 10), concurrency=1)
        chunker_reports[strategy] = report
        path = save_report(report, f"dense-{strategy}")
        report_paths[f"dense-{strategy}"] = str(path)
        print(
            f"dense {strategy}: chunks={count}; recall@10="
            f"{report['overall']['recall@10']:.3f}; mrr={report['overall']['mrr']:.3f}; "
            f"p95={report['overall']['latency_p95_s'] * 1000:.1f} ms; report={path}"
        )

    best_chunker = max(
        CHUNKERS,
        key=lambda name: (
            chunker_reports[name]["overall"]["recall@10"],
            chunker_reports[name]["overall"]["mrr"],
        ),
    )
    hybrid = make_hybrid_retriever(
        index, embedder.embed_query, strategy=best_chunker, dense_k=50, sparse_k=50,
    )
    hybrid_report = await evaluate(hybrid, questions, ks=(1, 5, 10), concurrency=1)
    path = save_report(hybrid_report, f"hybrid-{best_chunker}")
    report_paths["hybrid"] = str(path)
    print(
        f"hybrid {best_chunker}: recall@10={hybrid_report['overall']['recall@10']:.3f}; "
        f"mrr={hybrid_report['overall']['mrr']:.3f}; "
        f"p95={hybrid_report['overall']['latency_p95_s'] * 1000:.1f} ms; report={path}"
    )

    reranker = CrossEncoderReranker()
    candidate_count = 30
    reranked = make_reranked_hybrid_retriever(
        index,
        embedder.embed_query,
        reranker,
        strategy=best_chunker,
        candidate_k=candidate_count,
        dense_k=50,
        sparse_k=50,
    )
    reranked_report = await evaluate(reranked, questions, ks=(1, 5, 10), concurrency=1)
    path = save_report(reranked_report, f"hybrid-reranked-{best_chunker}-{candidate_count}")
    report_paths["hybrid_plus_rerank"] = str(path)
    print(
        f"hybrid + rerank {best_chunker} ({candidate_count} candidates): "
        f"recall@10={reranked_report['overall']['recall@10']:.3f}; "
        f"mrr={reranked_report['overall']['mrr']:.3f}; "
        f"p95={reranked_report['overall']['latency_p95_s'] * 1000:.1f} ms; report={path}"
    )

    ordered = {
        f"dense ({best_chunker})": chunker_reports[best_chunker],
        "hybrid": hybrid_report,
        "hybrid + rerank": reranked_report,
    }
    print("\nBest chunker:", best_chunker)
    print(format_comparison(ordered))
    pages, source_pages = count_pages()
    print(f"\nVerified questions: {len(questions)}; multi-hop: {sum(q.kind == 'multi_hop' for q in questions)}")
    print(f"Document pages: {pages} ({source_pages})")
    print("Saved reports:")
    for label, report_path in report_paths.items():
        print(f"  {label}: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
