"""Live check against Groq: concurrency, latency, token-count calibration, streaming.
Run:  python scripts/gateway_smoke.py
"""
import asyncio
import statistics
import time

from dotenv import load_dotenv

from aegisflow.llm_gateway import Message, from_env

N = 20
PROMPT = "List five facts about the Python language, one sentence each."
# gpt-oss is a reasoning model: reasoning tokens count against max_tokens.
OPTS = {"max_tokens": 600, "reasoning_effort": "low"}


async def main():
    load_dotenv()
    c = from_env(max_concurrency=5)
    msgs = [Message("user", PROMPT)]

    t0 = time.perf_counter()
    single = await c.complete(msgs, **OPTS)
    t_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = await asyncio.gather(*[c.complete(msgs, **OPTS) for _ in range(N)])
    t_batch = time.perf_counter() - t0

    print(f"single call:        {t_single:.2f}s  finish={single.finish_reason}")
    print(f"{N} concurrent calls: {t_batch:.2f}s  (serial would be ~{t_single * N:.1f}s)")
    print(f"retries used:       {sum(r.attempts - 1 for r in results)}")
    print(f"median latency:     {statistics.median(r.latency_s for r in results):.2f}s")
    print(f"empty responses:    {sum(1 for r in results if not r.text)}/{N}")

    # Token calibration: provider prompt_tokens - local estimate, 3 prompt sizes.
    gaps = []
    for reps in (1, 20, 80):
        m = [Message("user", "The quick brown fox jumps over the lazy dog. " * reps)]
        r = await c.complete(m, max_tokens=5, reasoning_effort="low")
        est = c.count_tokens(m)
        gaps.append(r.usage.prompt_tokens - est)
        print(f"calibration reps={reps:<3} estimated={est:<4} provider={r.usage.prompt_tokens:<4} gap={gaps[-1]}")
    print(f"suggested prompt_overhead={round(statistics.median(gaps))} (stable gap => fixed template cost)")

    chunks = [d async for d in c.stream(msgs, **OPTS)]
    print(f"stream:             {len(chunks)} chunks -> {''.join(chunks)[:60]!r}")
    await c.aclose()


asyncio.run(main())