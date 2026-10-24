"""Retrieval evaluation: golden-set I/O, metrics, harness, versioned reports."""
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

# async (query, k) -> ranked chunk ids, best first
Retriever = Callable[[str, int], Awaitable[list[str]]]


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    gold_chunk_ids: tuple[str, ...]
    kind: str = "single"  # "single" | "multi_hop"
    reference_answer: str | None = None
    verified: bool = False  # set True only after a human has checked it


def save_questions(questions: Iterable[GoldenQuestion], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({**asdict(q), "gold_chunk_ids": list(q.gold_chunk_ids)}) + "\n")


def load_questions(path: str | Path, only_verified: bool = False) -> list[GoldenQuestion]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["gold_chunk_ids"] = tuple(d["gold_chunk_ids"])
        q = GoldenQuestion(**d)
        if q.verified or not only_verified:
            out.append(q)
    return out


# ---- metrics ----------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        raise ValueError("gold set is empty")
    return len(gold & set(retrieved[:k])) / len(gold)


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    return float(bool(set(gold) & set(retrieved[:k])))


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    gold = set(gold)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def _pct(xs: Sequence[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)]


def _summarize(rows: list[dict], ks: Sequence[int]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        out[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / n
    out["mrr"] = sum(r["rr"] for r in rows) / n
    lat = [r["latency_s"] for r in rows]
    out["latency_p50_s"], out["latency_p95_s"] = _pct(lat, 50), _pct(lat, 95)
    return out


# ---- harness ----------------------------------------------------------------

async def evaluate(
    retriever: Retriever,
    questions: Sequence[GoldenQuestion],
    ks: Sequence[int] = (1, 5, 10),
    concurrency: int = 8,
) -> dict:
    if not questions:
        raise ValueError("no questions to evaluate")
    kmax, sem = max(ks), asyncio.Semaphore(concurrency)

    async def one(q: GoldenQuestion):
        async with sem:
            t0 = time.perf_counter()
            ids = await retriever(q.question, kmax)
            return ids, time.perf_counter() - t0

    results = await asyncio.gather(*[one(q) for q in questions])
    rows = []
    for q, (ids, latency) in zip(questions, results):
        row = {"id": q.id, "kind": q.kind, "latency_s": latency,
               "rr": reciprocal_rank(ids, q.gold_chunk_ids), "retrieved": ids[:kmax]}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ids, q.gold_chunk_ids, k)
            row[f"hit@{k}"] = hit_at_k(ids, q.gold_chunk_ids, k)
        rows.append(row)

    kinds = sorted({r["kind"] for r in rows})
    return {
        "ks": list(ks),
        "overall": _summarize(rows, ks),
        "by_kind": {kd: _summarize([r for r in rows if r["kind"] == kd], ks) for kd in kinds},
        "per_question": rows,
    }


def save_report(report: dict, name: str, results_dir: str | Path = "eval/results") -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = results_dir / f"{name}-{now.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"name": name, "created_at": now.isoformat(), **report}, indent=2))
    return path


def format_comparison(reports: dict[str, dict]) -> str:
    """Markdown ablation table; each column shows its delta vs the previous one."""
    names = list(reports)
    ks = reports[names[0]]["ks"]
    metrics = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr", "latency_p95_s"]
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in metrics:
        cells, prev = [], None
        for nm in names:
            v = reports[nm]["overall"][m]
            cells.append(f"{v:.3f}" if prev is None else f"{v:.3f} ({v - prev:+.3f})")
            prev = v
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def judge_agreement(scores: Sequence[float], human_faithful: Sequence[bool], threshold: float = 0.8) -> float:
    """Fraction of samples where (judge score >= threshold) matches the human label."""
    if len(scores) != len(human_faithful) or not scores:
        raise ValueError("need equal-length, non-empty inputs")
    return sum((s >= threshold) == h for s, h in zip(scores, human_faithful)) / len(scores)


# ---- human review -----------------------------------------------------------

def review_questions(questions, chunks_by_id, ask=input, show=print) -> list[GoldenQuestion]:
    """Interactive y/n/edit/quit pass. Kept questions are marked verified."""
    out = []
    for i, q in enumerate(questions):
        if q.verified:
            out.append(q)
            continue
        show(f"\n[{i + 1}/{len(questions)}] ({q.kind}) {q.question}\nANSWER: {q.reference_answer}")
        for cid in q.gold_chunk_ids:
            c = chunks_by_id.get(cid)
            show(f"--- {cid} ---\n{c.text[:600] if c else '(chunk text unavailable)'}")
        while True:
            a = ask("keep? [y]es / [n]o / [e]dit / [q]uit: ").strip().lower()
            if a == "y":
                out.append(replace(q, verified=True)); break
            if a == "n":
                break
            if a == "e":
                out.append(replace(q, question=ask("new question: ").strip(), verified=True)); break
            if a == "q":
                out.extend(questions[i:])
                return out
            show("?")
    return out