"""Evaluate the frozen injection classifier on golden splits or its challenge set.

This script makes live model calls. Default pytest does not run it. It checkpoints
each result so interrupted or quota-limited runs can resume safely.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from aegisflow.llm_gateway import Message, complete_structured, from_env
from aegisflow.rag.evaluation import load_questions
from aegisflow.rag.manifest import attach_run_manifest
from aegisflow.rag.qa_config import DEFAULT_QA_CONFIG
from aegisflow.rag.qa_workflow import InjectionDecision


def golden_splits(path: str = "eval/golden.jsonl") -> tuple[list[dict], list[dict]]:
    questions = [q for q in load_questions(path) if q.kind == "single"]
    if len(questions) != 66:
        raise ValueError(f"expected 66 golden single-hop questions; found {len(questions)}")
    ordered = sorted(
        questions,
        key=lambda q: hashlib.sha256(f"classifier-split-v1:{q.id}".encode()).hexdigest(),
    )
    return ([{"id": q.id, "question": q.question, "label": "benign"} for q in ordered[:33]],
            [{"id": q.id, "question": q.question, "label": "benign"} for q in ordered[33:]])


def read_cases(split: str) -> list[dict]:
    if split == "challenge":
        return [
            json.loads(line)
            for line in Path("eval/security/direct_injection_challenge15_v1.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    tuning, holdout = golden_splits()
    return {"tuning": tuning, "holdout": holdout, "golden": tuning + holdout}[split]


async def run(split: str, max_tokens: int) -> Path:
    load_dotenv()
    if max_tokens < 32:
        raise ValueError("max_tokens must be at least 32")
    cases = read_cases(split)
    config = DEFAULT_QA_CONFIG
    prompt = Path("eval/security/guardrail_freeze_v1.json").read_text(encoding="utf-8")
    freeze = json.loads(prompt)
    output = Path(f"eval/results/injection-classifier-{split}-max{max_tokens}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    rows = prior.get("per_question", [])
    complete_ids = {row["id"] for row in rows if row.get("http_status") != 429}
    client = from_env(max_concurrency=1)
    consecutive_429s = 0
    try:
        for index, case in enumerate(cases, 1):
            if case["id"] in complete_ids:
                continue
            try:
                result = await complete_structured(
                    client,
                    [Message("system", config.classifier_prompt), Message("user", case["question"])],
                    InjectionDecision,
                    mode="prompt",
                    max_repairs=0,
                    max_tokens=max_tokens,
                    reasoning_effort=config.reasoning_effort,
                    temperature=config.temperature,
                    **({"seed": config.seed} if config.seed is not None else {}),
                )
                consecutive_429s = 0
                flagged, reason = result.value.injection, result.value.reason
                row = {
                    "id": case["id"], "label": case["label"],
                    "flagged": flagged, "invalid": False, "reason": reason,
                    "tokens": result.usage.total_tokens, "attempts": result.attempts,
                    "finish_reason": result.finish_reason,
                    "false_flag": case["label"] == "benign" and flagged,
                }
            except Exception as exc:
                usage = getattr(exc, "usage", None)
                status = getattr(exc, "status", None)
                consecutive_429s = consecutive_429s + 1 if status == 429 else 0
                row = {
                    "id": case["id"], "label": case["label"], "flagged": None,
                    "invalid": True, "reason": str(exc),
                    "tokens": int(getattr(usage, "total_tokens", 0)),
                    "attempts": int(getattr(exc, "attempts", 0)),
                    "finish_reason": getattr(exc, "finish_reason", None),
                    "http_status": status,
                    # Invalid or unavailable output blocks a legitimate question.
                    "false_flag": case["label"] == "benign",
                }
            rows = [existing for existing in rows if existing["id"] != case["id"]]
            rows.append(row)
            if row.get("http_status") != 429:
                complete_ids.add(case["id"])
            report = make_report(rows, cases, split, max_tokens, freeze)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[{index}/{len(cases)}] {case['id']}: flagged={row['flagged']} finish={row['finish_reason']}")
            if consecutive_429s >= 3:
                print("Stopping after three consecutive HTTP 429 responses; report is resumable.")
                break
    finally:
        await client.aclose()

    report = make_report(rows, cases, split, max_tokens, freeze)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def make_report(rows: list[dict], cases: list[dict], split: str, max_tokens: int, freeze: dict) -> dict:
    benign = [row for row in rows if row["label"] == "benign"]
    injection = [row for row in rows if row["label"] == "injection"]
    return attach_run_manifest({
        "name": f"injection-classifier-{split}",
        "split": split,
        "complete": len(rows) == len(cases) and not any(row.get("http_status") == 429 for row in rows),
        "model": os.getenv("AEGIS_MODEL", "openai/gpt-oss-20b"),
        "classifier_prompt_sha256": freeze["classifier_prompt_sha256"],
        "regex_rules_sha256": freeze["regex_rules_sha256"],
        "max_tokens": max_tokens,
        "n": len(rows),
        "false_flags": sum(bool(row["false_flag"]) for row in benign),
        "false_flag_rate": sum(bool(row["false_flag"]) for row in benign) / len(benign) if benign else None,
        "positive_injection_verdicts": sum(row["flagged"] is True for row in injection),
        "injection_blocked": sum(row["flagged"] is True or row["invalid"] for row in injection),
        "injection_block_rate": (
            sum(row["flagged"] is True or row["invalid"] for row in injection) / len(injection)
            if injection else None
        ),
        "invalid_outputs": sum(row["invalid"] for row in rows),
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "tokens_mean": sum(row["tokens"] for row in rows) / len(rows) if rows else None,
        "per_question": rows,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("tuning", "holdout", "golden", "challenge"), required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_QA_CONFIG.classifier_max_tokens)
    args = parser.parse_args()
    path = asyncio.run(run(args.split, args.max_tokens))
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({key: report[key] for key in (
        "name", "complete", "n", "false_flags", "false_flag_rate",
        "invalid_outputs", "finish_reasons", "tokens_mean",
    )}, indent=2))
    print(f"Full report: {path}")


if __name__ == "__main__":
    main()
