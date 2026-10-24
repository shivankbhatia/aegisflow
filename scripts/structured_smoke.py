"""Live structured-extraction check: schema-valid rate, repairs, field accuracy.
Run:  python scripts/structured_smoke.py
"""
import asyncio
import random
import statistics
import time

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from aegisflow.llm_gateway import (
    Message, StructuredOutputError, complete_structured, from_env,
)

N = 30
CITIES = ["Austin", "Boston", "Denver", "Seattle", "Chicago", "Atlanta", "Phoenix", "Portland"]
NAMES = ["Acme", "Borealis", "Cobalt", "Delphi", "Ember", "Fjord", "Granite", "Helix"]


class FilingFacts(BaseModel):
    company: str = Field(description="Company name without suffix")
    fiscal_year: int
    revenue_usd_billions: float
    net_income_usd_billions: float
    hq_city: str


def make_doc(rng: random.Random):
    gold = FilingFacts(
        company=rng.choice(NAMES) + rng.choice(["tech", "works", "labs"]).capitalize(),
        fiscal_year=rng.randint(2015, 2024),
        revenue_usd_billions=round(rng.uniform(1, 90), 1),
        net_income_usd_billions=round(rng.uniform(0.1, 9), 1),
        hq_city=rng.choice(CITIES),
    )
    text = (
        f"{gold.company} Inc., headquartered in {gold.hq_city}, reported revenue of "
        f"${gold.revenue_usd_billions} billion for fiscal {gold.fiscal_year}, with net income of "
        f"${gold.net_income_usd_billions} billion. Management cited supply-chain headwinds."
    )
    return text, gold


async def main():
    load_dotenv()
    c = from_env(max_concurrency=5)
    rng = random.Random(7)
    docs = [make_doc(rng) for _ in range(N)]
    opts = {"max_tokens": 800, "reasoning_effort": "low"}

    async def run(text):
        try:
            return await complete_structured(
                c, [Message("user", f"Extract the facts from this filing excerpt:\n\n{text}")],
                FilingFacts, **opts)
        except StructuredOutputError as e:
            return e

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run(t) for t, _ in docs])
    elapsed = time.perf_counter() - t0

    ok = [(r, g) for r, (_, g) in zip(results, docs) if not isinstance(r, Exception)]
    fields = list(FilingFacts.model_fields)
    correct = sum(getattr(r.value, f) == getattr(g, f) for r, g in ok for f in fields)
    print(f"docs: {N}  elapsed: {elapsed:.1f}s")
    print(f"schema-valid:       {len(ok)}/{N} = {len(ok) / N:.0%}")
    print(f"mean repairs/call:  {statistics.mean(r.attempts - 1 for r, _ in ok):.2f}" if ok else "no valid results")
    print(f"field accuracy:     {correct}/{len(ok) * len(fields)} = {correct / max(len(ok) * len(fields), 1):.0%}")
    print(f"tokens (all calls): {sum(r.usage.total_tokens for r, _ in ok)}")
    for r in results:
        if isinstance(r, Exception):
            print("FAILED:", r, "| last raw:", r.last_raw[:120])
    await c.aclose()


asyncio.run(main())