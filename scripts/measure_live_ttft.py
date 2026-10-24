"""Measure live /query time-to-first-token without running a judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


async def measure(samples: int, query: str, k: int, api_url: str) -> Path:
    load_dotenv()
    rows: list[dict] = []
    timeout = httpx.Timeout(float(os.getenv("AEGIS_EVAL_TIMEOUT_S", "180")))
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index in range(1, samples + 1):
            done = None
            error = None
            async with client.stream(
                "POST", f"{api_url.rstrip('/')}/query", json={"query": query, "k": k}
            ) as response:
                response.raise_for_status()
                event_name = None
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:") and event_name:
                        payload = json.loads(line[5:].strip())
                        if event_name == "done":
                            done = payload
                        elif event_name == "error":
                            error = payload
                        event_name = None
            if error:
                raise RuntimeError(
                    f"Sample {index} failed; stopping to avoid repeated retries/calls: "
                    f"{error.get('detail')}"
                )
            if not done or done.get("timings_ms", {}).get("first_token") is None:
                raise RuntimeError(f"Sample {index} completed without first-token timing")
            row = {
                "sample": index,
                "request_id": done["request_id"],
                "first_token_ms": float(done["timings_ms"]["first_token"]),
                "retrieval_ms": float(done["timings_ms"]["retrieval"]),
                "rerank_ms": float(done["timings_ms"]["rerank"]),
                "total_ms": float(done["timings_ms"]["total"]),
                "tokens": done["tokens"],
            }
            rows.append(row)
            print(
                f"[{index}/{samples}] first_token={row['first_token_ms']:.1f} ms "
                f"retrieval={row['retrieval_ms']:.1f} ms rerank={row['rerank_ms']:.1f} ms"
            )

    values = [row["first_token_ms"] for row in rows]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "query": query,
        "k": k,
        "sample_count": len(values),
        "percentile_method": "nearest rank",
        "p50_first_token_ms": percentile(values, 0.50),
        "p95_first_token_ms": percentile(values, 0.95),
        "target_ms": 1500,
        "target_passed": percentile(values, 0.95) < 1500,
        "samples": rows,
    }
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"live-ttft-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "p50_first_token_ms", "p95_first_token_ms", "target_ms", "target_passed"
    )}, indent=2))
    print(f"Full report: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--query", default="How do I add a background task?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--api-url", default=os.getenv("AEGIS_API_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    if args.samples < 1 or args.k < 1:
        parser.error("--samples and --k must be positive")
    asyncio.run(measure(args.samples, args.query, args.k, args.api_url))


if __name__ == "__main__":
    main()
