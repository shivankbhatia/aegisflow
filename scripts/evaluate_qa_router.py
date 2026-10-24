"""Score the authored 30-query router set and save per-query judgments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.llm_gateway import from_env
from aegisflow.rag.qa_workflow import _ROUTER_PROMPT, route_question
from aegisflow.rag.manifest import attach_run_manifest


async def run() -> Path:
    load_dotenv()
    cases = [json.loads(line) for line in Path("eval/router_test20.jsonl").read_text().splitlines() if line]
    clean_cases = [case for case in cases if case["test_eligible"]]
    if len(cases) != 20 or len(clean_cases) != 19:
        raise RuntimeError("router set must contain 20 rows, with 19 uncontaminated cases")

    client = from_env(max_concurrency=1)
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "qa-router-test-v2.json"
    rows = json.loads(output.read_text(encoding="utf-8")).get("per_query", []) if output.exists() else []
    completed = {row["id"] for row in rows}
    try:
        for index, case in enumerate(cases, 1):
            if not case["test_eligible"]:
                continue
            if case["id"] in completed:
                continue
            started = time.perf_counter()
            predicted, tokens, attempts = await route_question(client, case["question"])
            row = {
                **case,
                "predicted_route": predicted,
                "correct": predicted == case["route"],
                "tokens": tokens,
                "latency_s": time.perf_counter() - started,
                "attempts": attempts,
            }
            rows.append(row)
            completed.add(row["id"])
            output.write_text(json.dumps(make_report(rows, complete=False), indent=2), encoding="utf-8")
            print(f"[{index}/20] {case['id']}: expected={case['route']} predicted={predicted}")
    finally:
        await client.aclose()

    report = make_report(rows, complete=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("accuracy", "target_accuracy", "target_met", "by_route")}, indent=2))
    print(f"Full report: {output}")
    return output


def make_report(rows: list[dict], *, complete: bool) -> dict:
    test_cases = [json.loads(line) for line in Path("eval/router_test20.jsonl").read_text().splitlines() if line]
    contaminated_count = sum(not case["test_eligible"] for case in test_cases)
    by_route = {}
    for route in ("direct", "multi_hop", "out_of_scope"):
        subset = [row for row in rows if row["route"] == route]
        by_route[route] = {
            "correct": sum(row["correct"] for row in subset),
            "total": len(subset),
            "accuracy": sum(row["correct"] for row in subset) / len(subset) if subset else None,
        }
    accuracy = sum(row["correct"] for row in rows) / len(rows) if rows else None
    return attach_run_manifest({
        "name": "qa-router-authored-set-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "authorship": "Codex-authored, labels assigned by intended query scope",
        "calibration": "Internal authored set; not human-labeled or independently calibrated",
        "prompt_version": "v2-frozen",
        "prompt_sha256": hashlib.sha256(_ROUTER_PROMPT.encode()).hexdigest(),
        "model": os.getenv("AEGIS_MODEL", "openai/gpt-oss-20b"),
        "target_count": 20,
        "test_set_count": 20,
        "contaminated_rows_excluded": contaminated_count,
        "clean_test_count": len(test_cases) - contaminated_count,
        "n": len(rows),
        "accuracy": accuracy,
        "target_accuracy": 0.90,
        "target_met": accuracy is not None and accuracy >= 0.90,
        "by_route": by_route,
        "per_query": rows,
    })


if __name__ == "__main__":
    asyncio.run(run())
