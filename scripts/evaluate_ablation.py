"""Run the final dense/chunker, hybrid, and reranker retrieval ablation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.rag.corpus import iter_documents
from aegisflow.rag.embeddings import BGEEmbedder
from aegisflow.rag.evaluation import (
    document_level_view, evaluate, load_questions, paired_bootstrap, save_report,
)
from aegisflow.rag.reranker import CrossEncoderReranker, make_reranked_hybrid_retriever
from aegisflow.rag.vector_index import PgVectorIndex, make_hybrid_retriever, make_retriever


SOURCES = {
    "langchain": Path("data/raw/langchain-docs/src/oss"),
    "fastapi": Path("data/raw/fastapi/docs/en/docs"),
    "psycopg": Path("data/raw/psycopg/docs"),
}
CHUNKERS = ("fixed-size", "structure-aware", "semantic")
MODES = ("dense", "hybrid", "hybrid_rerank")


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
    report_paths: dict[str, str] = {}

    reranker = CrossEncoderReranker()
    reports: dict[tuple[str, str], dict] = {}
    report_paths: dict[tuple[str, str], str] = {}
    for strategy in CHUNKERS:
        if index.count(strategy) == 0:
            raise RuntimeError(f"No indexed chunks for strategy {strategy!r}")
        dense = make_retriever(index, embedder.embed_query, strategy=strategy)
        hybrid = make_hybrid_retriever(index, embedder.embed_query, strategy=strategy)
        reranked = make_reranked_hybrid_retriever(
            index, embedder.embed_query, reranker, strategy=strategy, candidate_k=5,
        )
        retrievers = {"dense": dense, "hybrid": hybrid, "hybrid_rerank": reranked}
        for mode in MODES:
            base = retrievers[mode]
            if mode == "hybrid_rerank":
                async def retrieve(query, k, base=base):
                    return await base(query, k, candidate_limit=k)
            else:
                async def retrieve(query, k, base=base):
                    return await base(query, k)
            doc_retriever, doc_questions = document_level_view(
                retrieve, questions, chunk_fetch_limit=5,
            )
            report = await evaluate(doc_retriever, doc_questions, ks=(5,), concurrency=1)
            name = f"served-doc-{mode}-{strategy}-k5"
            path = save_report(report, name)
            reports[(mode, strategy)] = report
            report_paths[(mode, strategy)] = str(path)
            print(
                f"{name}: document recall@5={report['overall']['recall@5']:.3f}; "
                f"p95={report['overall']['latency_p95_s'] * 1000:.1f} ms; report={path}"
            )

    # Compare each of the nine configurations with the incumbent before
    # selecting the best strategy for the larger reranker-pool experiment.
    incumbent_report = reports[("dense", "structure-aware")]
    bootstraps = {}
    for (mode, strategy), report in reports.items():
        result = paired_bootstrap(
            incumbent_report, report, "recall@5", seed=20261024,
        )
        bootstraps[f"{mode}/{strategy}"] = result
        print(
            f"bootstrap vs dense/structure-aware {mode}/{strategy}: "
            f"delta={result['mean_difference']:+.3f}, "
            f"95% CI=[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}], "
            f"better/worse={result['better']}/{result['worse']}"
        )
    bootstrap_path = Path("eval/results/served-doc-bootstrap-20261024.json")
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")

    feasible = [
        key for key, report in reports.items()
        if report["overall"]["latency_p95_s"] < 0.150
    ]
    candidate = max(feasible, key=lambda key: reports[key]["overall"]["recall@5"])
    candidate_result = bootstraps[f"{candidate[0]}/{candidate[1]}"]
    if candidate_result["ci_low"] <= 0 <= candidate_result["ci_high"]:
        best_mode, best_chunker = "dense", "structure-aware"
    else:
        best_mode, best_chunker = candidate

    # Compare 15 and 30 candidates for the selected strategy; final output k=5
    # is the same as the served endpoint.
    pool_retrievers = {}
    for candidate_count in (15, 30):
        reranked = make_reranked_hybrid_retriever(
            index, embedder.embed_query, reranker, strategy=best_chunker,
            candidate_k=candidate_count,
        )
        async def retrieve_pool(query, k, base=reranked, candidate_limit=candidate_count):
            return await base(query, k, candidate_limit=candidate_limit)
        doc_retriever, doc_questions = document_level_view(
            retrieve_pool, questions, chunk_fetch_limit=5,
        )
        report = await evaluate(doc_retriever, doc_questions, ks=(5,), concurrency=1)
        name = f"served-doc-hybrid-rerank-{best_chunker}-k5-pool{candidate_count}"
        path = save_report(report, name)
        pool_retrievers[candidate_count] = str(path)
        print(
            f"{name}: document recall@5={report['overall']['recall@5']:.3f}; "
            f"p95={report['overall']['latency_p95_s'] * 1000:.1f} ms; report={path}"
        )

    print(f"\nDecision rule selects: {best_mode}/{best_chunker}")
    pages, source_pages = count_pages()
    print(f"\nVerified questions: {len(questions)}; multi-hop: {sum(q.kind == 'multi_hop' for q in questions)}")
    print(f"Document pages: {pages} ({source_pages})")
    print(f"Bootstrap details: {bootstrap_path}")
    print("Pool reports:", pool_retrievers)


if __name__ == "__main__":
    asyncio.run(main())
