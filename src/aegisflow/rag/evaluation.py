"""Retrieval evaluation: golden-set I/O, metrics, harness, versioned reports."""
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Mapping, Sequence

from .manifest import make_run_manifest, validate_run_manifest

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
    reviewed_by: str | None = None  # records delegated assistant review transparently


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
        "manifest": make_run_manifest(),
        "ks": list(ks),
        "overall": _summarize(rows, ks),
        "by_kind": {kd: _summarize([r for r in rows if r["kind"] == kd], ks) for kd in kinds},
        "per_question": rows,
    }


def document_level_view(
    retriever: Retriever,
    questions: Sequence[GoldenQuestion],
    *,
    chunk_fetch_limit: int | None = None,
) -> tuple[Retriever, list[GoldenQuestion]]:
    """Compare chunking strategies using document identity, not colliding chunk IDs."""
    def document_id(chunk_id: str) -> str:
        return chunk_id.split("#", 1)[0]

    async def retrieve_documents(query: str, k: int) -> list[str]:
        # Over-fetch chunks so repeated hits from one document do not leave
        # document-level top-k artificially short.
        fetch_k = chunk_fetch_limit if chunk_fetch_limit is not None else max(k * 10, 50)
        ids = await retriever(query, fetch_k)
        # Keep the first (best ranked) hit for each document, up to k docs.
        return list(dict.fromkeys(document_id(chunk_id) for chunk_id in ids))[:k]

    doc_questions = [
        replace(q, gold_chunk_ids=tuple(dict.fromkeys(document_id(cid) for cid in q.gold_chunk_ids)))
        for q in questions
    ]
    return retrieve_documents, doc_questions


def paired_bootstrap(
    report_a: str | Path | Mapping,
    report_b: str | Path | Mapping,
    metric: str,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Compare two reports on matched questions; differences are B minus A."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    def read_report(report):
        return json.loads(Path(report).read_text(encoding="utf-8")) if isinstance(report, (str, Path)) else report

    a, b = read_report(report_a), read_report(report_b)
    key = "rr" if metric == "mrr" else metric

    def values_by_id(report):
        rows = report.get("per_question", [])
        values = {}
        for row in rows:
            if row["id"] in values:
                raise ValueError(f"duplicate question id: {row['id']}")
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError(f"metric {metric!r} missing or non-finite for {row.get('id')}")
            values[row["id"]] = float(row[key])
        return values

    av, bv = values_by_id(a), values_by_id(b)
    ids = sorted(av.keys() & bv.keys())
    if not ids:
        raise ValueError("reports have no matching question IDs")
    differences = [bv[qid] - av[qid] for qid in ids]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    low_i = max(0, math.floor(tail * resamples))
    high_i = min(resamples - 1, math.ceil((1 - tail) * resamples) - 1)
    return {
        "metric": metric,
        "difference": "report_b - report_a",
        "n": n,
        "mean_difference": statistics.fmean(differences),
        "confidence": confidence,
        "ci_low": samples[low_i],
        "ci_high": samples[high_i],
        "better": sum(d > 0 for d in differences),
        "worse": sum(d < 0 for d in differences),
        "tied": sum(d == 0 for d in differences),
        "unmatched_a": sorted(av.keys() - bv.keys()),
        "unmatched_b": sorted(bv.keys() - av.keys()),
        "resamples": resamples,
        "seed": seed,
    }


def save_report(report: dict, name: str, results_dir: str | Path = "eval/results") -> Path:
    if "manifest" not in report:
        raise ValueError("evaluation report must include a run manifest")
    validate_run_manifest(report["manifest"])
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
