import json
import pytest

from aegisflow.rag.evaluation import (
    GoldenQuestion, evaluate, format_comparison, hit_at_k, judge_agreement,
    load_questions, paired_bootstrap, recall_at_k, reciprocal_rank, review_questions,
    save_questions, save_report,
)
from aegisflow.rag.types import Chunk


def test_metrics_basic_and_edges():
    ret = ["a", "b", "c", "d"]
    assert recall_at_k(ret, {"c"}, 2) == 0.0 and recall_at_k(ret, {"c"}, 3) == 1.0
    assert recall_at_k(ret, {"a", "z"}, 10) == 0.5          # multi-hop: partial credit
    assert hit_at_k(ret, {"a", "z"}, 1) == 1.0 and hit_at_k(ret, {"z"}, 10) == 0.0
    assert reciprocal_rank(ret, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ret, {"b", "d"}) == 0.5          # first hit counts
    assert reciprocal_rank(ret, {"z"}) == 0.0
    assert recall_at_k([], {"a"}, 5) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(ret, set(), 3)


QS = [
    GoldenQuestion("s1", "q about a", ("a",)),
    GoldenQuestion("s2", "q about b", ("b",)),
    GoldenQuestion("m1", "q about a and c", ("a", "c"), kind="multi_hop"),
]
INDEX = {"q about a": ["a", "x"], "q about b": ["x", "b"], "q about a and c": ["c", "x", "a"]}


async def fake_retriever(query, k):
    return INDEX[query][:k]


@pytest.mark.asyncio
async def test_evaluate_aggregates_overall_and_by_kind():
    rep = await evaluate(fake_retriever, QS, ks=(1, 2, 3))
    o = rep["overall"]
    assert o["n"] == 3
    assert o["recall@1"] == pytest.approx((1 + 0 + 0.5) / 3)
    assert o["recall@3"] == pytest.approx(1.0)
    assert o["mrr"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert rep["by_kind"]["multi_hop"]["n"] == 1 and rep["by_kind"]["single"]["n"] == 2
    assert rep["by_kind"]["multi_hop"]["recall@1"] == 0.5
    assert o["latency_p95_s"] >= o["latency_p50_s"] >= 0
    assert [r["id"] for r in rep["per_question"]] == ["s1", "s2", "m1"]  # order preserved


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_and_roundtrips_report(tmp_path):
    with pytest.raises(ValueError):
        await evaluate(fake_retriever, [])
    rep = await evaluate(fake_retriever, QS, ks=(1, 3))
    path = save_report(rep, "dense", tmp_path)
    data = json.loads(path.read_text())
    assert path.name.startswith("dense-") and data["name"] == "dense" and data["overall"]["n"] == 3


def test_questions_jsonl_roundtrip_and_verified_filter(tmp_path):
    qs = [GoldenQuestion("a", "q1", ("c1",), reference_answer="r", verified=True),
          GoldenQuestion("b", "q2", ("c1", "c2"), kind="multi_hop")]
    p = tmp_path / "sub" / "golden.jsonl"
    save_questions(qs, p)
    assert load_questions(p) == qs
    assert [q.id for q in load_questions(p, only_verified=True)] == ["a"]


def test_format_comparison_shows_deltas():
    def rep(r1, mrr, lat):
        return {"ks": [1], "overall": {"recall@1": r1, "hit@1": r1, "mrr": mrr, "latency_p95_s": lat}}
    table = format_comparison({"dense": rep(0.5, 0.6, 0.05), "hybrid": rep(0.6, 0.7, 0.07)})
    assert "| metric | dense | hybrid |" in table
    assert "| recall@1 | 0.500 | 0.600 (+0.100) |" in table


def test_judge_agreement():
    assert judge_agreement([0.9, 0.2, 0.85, 0.5], [True, False, False, False]) == 0.75
    with pytest.raises(ValueError):
        judge_agreement([], [])


def test_paired_bootstrap_matches_ids_and_reports_known_differences(tmp_path):
    a = {"per_question": [
        {"id": "q1", "recall@5": 0.0},
        {"id": "q2", "recall@5": 1.0},
        {"id": "q3", "recall@5": 0.0},
    ]}
    b = {"per_question": [
        {"id": "q3", "recall@5": 1.0},
        {"id": "q1", "recall@5": 1.0},
        {"id": "q2", "recall@5": 1.0},
    ]}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))

    result = paired_bootstrap(pa, pb, "recall@5", resamples=2_000, seed=7)

    assert result["n"] == 3
    assert result["mean_difference"] == pytest.approx(2 / 3)
    assert (result["better"], result["worse"], result["tied"]) == (2, 0, 1)
    assert result["ci_low"] <= result["mean_difference"] <= result["ci_high"]
    assert result["unmatched_a"] == result["unmatched_b"] == []


def test_review_questions_yes_no_edit_quit():
    chunks = {"c1": Chunk("c1", "d", "text one")}
    qs = [GoldenQuestion(f"q{i}", f"question {i}", ("c1",), reference_answer="a") for i in range(5)]
    answers = iter(["?", "y", "n", "e", "rewritten", "q"])
    shown = []
    out = review_questions(qs, chunks, ask=lambda _: next(answers), show=shown.append)
    assert [q.id for q in out] == ["q0", "q2", "q3", "q4"]   # q1 dropped; q3,q4 kept via quit
    assert out[0].verified and out[1].verified and out[1].question == "rewritten"
    assert not out[2].verified and not out[3].verified
    assert any("text one" in s for s in shown)
