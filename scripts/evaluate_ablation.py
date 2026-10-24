"""Run the final dense/chunker, hybrid, and reranker retrieval ablation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.corpus import iter_documents
from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import (
    document_level_view, evaluate, format_comparison, load_questions, save_report,
)
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
        doc_retriever, doc_questions = document_level_view(retriever, questions)
        report = await evaluate(doc_retriever, doc_questions, ks=(1, 5, 10), concurrency=1)
        chunker_reports[strategy] = report
        path = save_report(report, f"doc-level-dense-{strategy}")
        report_paths[f"doc-level-dense-{strategy}"] = str(path)
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
    # Evaluate the served configuration exactly: the API limits the fused
    # candidate set to request.k (5 by default), then reranks those candidates.
    reranker = CrossEncoderReranker()
    served_strategy = "structure-aware"
    dense_top5 = make_retriever(index, embedder.embed_query, strategy=served_strategy)
    dense_top5_report = await evaluate(dense_top5, questions, ks=(5,), concurrency=1)
    path = save_report(dense_top5_report, f"served-baseline-dense-{served_strategy}-k5")
    report_paths["served_dense_k5"] = str(path)

    hybrid_top5 = make_hybrid_retriever(
        index, embedder.embed_query, strategy=served_strategy, dense_k=50, sparse_k=50,
    )
    hybrid_top5_report = await evaluate(hybrid_top5, questions, ks=(5,), concurrency=1)
    path = save_report(hybrid_top5_report, f"served-hybrid-{served_strategy}-k5")
    report_paths["served_hybrid_k5"] = str(path)

    reranked_served = make_reranked_hybrid_retriever(
        index,
        embedder.embed_query,
        reranker,
        strategy=served_strategy,
        candidate_k=30,
        dense_k=50,
        sparse_k=50,
    )

    async def served_retrieve(query: str, k: int) -> list[str]:
        return await reranked_served(query, k, candidate_limit=k)

    served_report = await evaluate(served_retrieve, questions, ks=(5,), concurrency=1)
    path = save_report(served_report, f"served-hybrid-rerank-{served_strategy}-k5")
    report_paths["served_hybrid_rerank_k5"] = str(path)
    print(
        f"served API config k=5: dense/hybrid/hybrid+rerank recall@5="
        f"{dense_top5_report['overall']['recall@5']:.3f}/"
        f"{hybrid_top5_report['overall']['recall@5']:.3f}/"
        f"{served_report['overall']['recall@5']:.3f}; MRR="
        f"{dense_top5_report['overall']['mrr']:.3f}/"
        f"{hybrid_top5_report['overall']['mrr']:.3f}/"
        f"{served_report['overall']['mrr']:.3f}"
    )

    print("\nBest chunker by document-level recall@10 then MRR:", best_chunker)
    print("Document-level chunker comparison:")
    print(format_comparison({name: chunker_reports[name] for name in CHUNKERS}))
    pages, source_pages = count_pages()
    print(f"\nVerified questions: {len(questions)}; multi-hop: {sum(q.kind == 'multi_hop' for q in questions)}")
    print(f"Document pages: {pages} ({source_pages})")
    print("Saved reports:")
    for label, report_path in report_paths.items():
        print(f"  {label}: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
