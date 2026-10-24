"""Run live answer, citation, latency, and faithfulness checks over verified goldens."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from aegisflow.llm_gateway import from_env
from aegisflow.rag.evaluation import load_questions
from aegisflow.rag.llm_eval import judge_faithfulness
from aegisflow.rag.vector_index import PgVectorIndex


MARKER = re.compile(r"\[(\d+)\]")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


async def run() -> Path:
    load_dotenv()
    questions = load_questions(os.getenv("AEGIS_GOLDEN_SET", "eval/golden.jsonl"), only_verified=True)
    if not questions:
        raise RuntimeError("No verified questions found")

    api_url = os.getenv("AEGIS_API_URL", "http://127.0.0.1:8000").rstrip("/")
    timeout = httpx.Timeout(float(os.getenv("AEGIS_EVAL_TIMEOUT_S", "180")))
    client = from_env(max_concurrency=1)
    index = PgVectorIndex(os.environ["DATABASE_URL"])
    rows = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            for number, question in enumerate(questions, start=1):
                started = time.perf_counter()
                answer_parts: list[str] = []
                citations_event: dict = {}
                first_token_s: float | None = None
                async with http.stream(
                    "POST", f"{api_url}/query", json={"query": question.question, "k": 10}
                ) as response:
                    response.raise_for_status()
                    event_name = None
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:") and event_name:
                            payload = json.loads(line[5:].strip())
                            if event_name == "token":
                                if first_token_s is None:
                                    first_token_s = time.perf_counter() - started
                                answer_parts.append(payload.get("text", ""))
                            elif event_name == "citations":
                                citations_event = payload
                            elif event_name == "error":
                                raise RuntimeError(f"Query {question.id} failed: {payload.get('detail')}")
                            event_name = None

                answer = "".join(answer_parts)
                markers = [int(value) for value in MARKER.findall(answer)]
                invalid = set(citations_event.get("invalid_markers", []))
                valid_marker_count = sum(marker not in invalid for marker in markers)
                citation_ids = list(dict.fromkeys(
                    item["chunk_id"] for item in citations_event.get("citations", [])
                ))
                chunks = await asyncio.to_thread(
                    index.fetch_chunks, citation_ids, strategy="structure-aware"
                )
                contexts = [chunks[cid]["text"] for cid in citation_ids if cid in chunks]
                judgement = await judge_faithfulness(client, question.question, answer, contexts)
                rows.append({
                    "id": question.id,
                    "kind": question.kind,
                    "answer": answer,
                    "first_token_s": first_token_s,
                    "faithfulness": judgement.score,
                    "supported_claims": judgement.verdict.supported_claims,
                    "unsupported_claims": judgement.verdict.unsupported_claims,
                    "unsupported": judgement.verdict.unsupported,
                    "citation_markers": len(markers),
                    "valid_citation_markers": valid_marker_count,
                    "invalid_markers": sorted(invalid),
                    "citation_chunk_ids": citation_ids,
                })
                print(f"[{number}/{len(questions)}] {question.id}: faithfulness={judgement.score:.2f}, citations={valid_marker_count}/{len(markers)}")
    finally:
        await client.aclose()

    total_markers = sum(row["citation_markers"] for row in rows)
    valid_markers = sum(row["valid_citation_markers"] for row in rows)
    latencies = [row["first_token_s"] for row in rows if row["first_token_s"] is not None]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "questions": len(rows),
        "verified_questions": len(questions),
        "multi_hop_questions": sum(q.kind == "multi_hop" for q in questions),
        "mean_faithfulness": sum(row["faithfulness"] for row in rows) / len(rows),
        "citation_validity": valid_markers / total_markers if total_markers else 0.0,
        "citation_markers": total_markers,
        "answers_with_citations": sum(row["citation_markers"] > 0 for row in rows),
        "time_to_first_token_p50_s": percentile(latencies, 0.50) if latencies else None,
        "time_to_first_token_p95_s": percentile(latencies, 0.95) if latencies else None,
        "targets": {"faithfulness": 0.85, "citation_validity": 0.95, "ttft_s": 1.5},
        "passed": {
            "faithfulness": (sum(row["faithfulness"] for row in rows) / len(rows)) >= 0.85,
            "citation_validity": (valid_markers / total_markers if total_markers else 0.0) >= 0.95,
            "ttft_p95": bool(latencies) and percentile(latencies, 0.95) < 1.5,
        },
        "per_question": rows,
    }
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"answers-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "questions", "multi_hop_questions", "mean_faithfulness", "citation_validity",
        "citation_markers", "answers_with_citations", "time_to_first_token_p50_s",
        "time_to_first_token_p95_s", "passed",
    )}, indent=2))
    print(f"Full report: {output}")
    return output


if __name__ == "__main__":
    asyncio.run(run())
