import json

import httpx
import pytest
from pydantic import BaseModel

from aegisflow.llm_gateway import (
    Message, OpenAICompatClient, StructuredOutputError, complete_structured,
)

MSGS = [Message("user", "extract")]


class Facts(BaseModel):
    company: str
    year: int


def body(text, finish="stop", p=10, c=5):
    return {"model": "m", "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": p, "completion_tokens": c}}


def scripted(replies):
    """Client whose Nth call returns replies[N]; records request bodies."""
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        r = replies[min(len(seen) - 1, len(replies) - 1)]
        return r(req) if callable(r) else httpx.Response(200, json=body(r))

    async def nosleep(d):
        pass

    c = OpenAICompatClient(api_key="k", base_url="https://x.test/v1", model="m",
                           transport=httpx.MockTransport(handler), sleep=nosleep)
    return c, seen


@pytest.mark.asyncio
async def test_valid_first_try_and_sends_json_schema():
    c, seen = scripted(['{"company":"Acme","year":2023}'])
    r = await complete_structured(c, MSGS, Facts)
    assert r.value == Facts(company="Acme", year=2023) and r.attempts == 1
    rf = seen[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "Facts"
    assert "company" in rf["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_fenced_json_is_accepted():
    c, _ = scripted(['```json\n{"company":"Acme","year":2023}\n```'])
    assert (await complete_structured(c, MSGS, Facts)).value.year == 2023


@pytest.mark.asyncio
async def test_repair_loop_feeds_back_error_and_aggregates_usage():
    c, seen = scripted(['{"company":"Acme"}', '{"company":"Acme","year":2023}'])
    r = await complete_structured(c, MSGS, Facts)
    assert r.attempts == 2 and r.usage.total_tokens == 30
    second = seen[1]["messages"]
    assert len(second) == len(seen[0]["messages"]) + 2
    assert second[-2]["role"] == "assistant" and '{"company":"Acme"}' in second[-2]["content"]
    assert "year" in second[-1]["content"]  # names the failing field


@pytest.mark.asyncio
async def test_exhausted_repairs_raise_with_last_output():
    c, seen = scripted(["not json"])
    with pytest.raises(StructuredOutputError) as e:
        await complete_structured(c, MSGS, Facts, max_repairs=2)
    assert e.value.attempts == 3 and e.value.last_raw == "not json" and len(seen) == 3


@pytest.mark.asyncio
async def test_truncation_raises_immediately_without_repair():
    c, seen = scripted([lambda r: httpx.Response(200, json=body("", finish="length"))])
    with pytest.raises(StructuredOutputError, match="truncated"):
        await complete_structured(c, MSGS, Facts)
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_prompt_mode_injects_schema_and_extracts_from_prose():
    c, seen = scripted(['Sure! {"company":"Acme","year":2023} hope that helps'])
    r = await complete_structured(c, MSGS, Facts, mode="prompt")
    assert r.value.company == "Acme"
    assert "response_format" not in seen[0]
    assert seen[0]["messages"][0]["role"] == "system" and "JSON Schema" in seen[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_falls_back_to_prompt_mode_when_response_format_rejected():
    def reject(req):
        if "response_format" in json.loads(req.content):
            return httpx.Response(400, text="response_format unsupported")
        return httpx.Response(200, json=body('{"company":"Acme","year":1}'))

    c, _ = scripted([reject])
    r = await complete_structured(c, MSGS, Facts)
    assert r.value.year == 1