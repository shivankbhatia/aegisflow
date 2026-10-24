import json

import httpx
import pytest

from aegisflow.llm_gateway import OpenAICompatClient
from aegisflow.rag.llm_eval import generate_questions, judge_faithfulness
from aegisflow.rag.types import Chunk


def client_for(reply_fn):
    prompts = []

    def handler(req):
        # whole conversation, so repair retries still "see" the original passages
        prompt = "\n".join(m["content"] for m in json.loads(req.content)["messages"])
        prompts.append(prompt)
        text = reply_fn(prompt)
        return httpx.Response(200, json={
            "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    async def nosleep(d):
        pass

    c = OpenAICompatClient(api_key="k", base_url="https://x.test/v1", model="m",
                           transport=httpx.MockTransport(handler), sleep=nosleep)
    return c, prompts


A, B, C = (Chunk(i, "doc", f"text of chunk {i}") for i in "abc")


@pytest.mark.asyncio
async def test_generate_single_and_multi_hop_and_skips_failures():
    def reply(prompt):
        if "chunk c" in prompt:
            return "not json at all"          # unrecoverable -> skipped
        return json.dumps({"question": " What is in chunk? ", "reference_answer": " stuff "})

    c, prompts = client_for(reply)
    single = await generate_questions(c, [[A], [B], [C]], kind="single", id_prefix="s")
    assert [q.id for q in single] == ["s000", "s001"]          # C skipped, order kept
    assert single[0].gold_chunk_ids == ("a",) and single[0].kind == "single"
    assert single[0].question == "What is in chunk?" and single[0].reference_answer == "stuff"
    assert single[0].verified is False

    prompts.clear()
    multi = await generate_questions(c, [[A, B]], kind="multi_hop", id_prefix="m")
    assert multi[0].gold_chunk_ids == ("a", "b") and multi[0].kind == "multi_hop"
    assert "[Passage 1]" in prompts[0] and "[Passage 2]" in prompts[0] and "ALL the passages" in prompts[0]


@pytest.mark.asyncio
async def test_judge_scores_supported_fraction_and_handles_no_claims():
    c, _ = client_for(lambda p: json.dumps({"supported_claims": 3, "unsupported_claims": 1, "unsupported": ["x"]}))
    j = await judge_faithfulness(c, "q", "ans", ["ctx"])
    assert j.score == 0.75 and j.verdict.unsupported == ["x"]

    c, _ = client_for(lambda p: json.dumps({"supported_claims": 0, "unsupported_claims": 0}))
    assert (await judge_faithfulness(c, "q", "I don't know", ["ctx"])).score == 1.0