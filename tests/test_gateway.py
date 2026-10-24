import asyncio
import random

import httpx
import pytest

from aegisflow.llm_gateway import LLMClient, LLMError, Message, OpenAICompatClient

MSGS = [Message("user", "hi")]


def ok_body(text="hello"):
    return {
        "model": "m",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def make_client(handler, **kw):
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    c = OpenAICompatClient(
        api_key="k",
        base_url="https://x.test/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        **kw,
    )
    return c, delays


def test_satisfies_protocol():
    c, _ = make_client(lambda r: httpx.Response(200, json=ok_body()))
    assert isinstance(c, LLMClient)


@pytest.mark.asyncio
async def test_complete_parses_text_and_usage():
    c, _ = make_client(lambda r: httpx.Response(200, json=ok_body("yo")))
    r = await c.complete(MSGS)
    assert (r.text, r.usage.prompt_tokens, r.usage.completion_tokens) == ("yo", 5, 2)
    assert r.usage.total_tokens == 7 and r.attempts == 1


@pytest.mark.asyncio
async def test_retries_429_and_honors_retry_after():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json=ok_body())

    c, delays = make_client(handler)
    r = await c.complete(MSGS)
    assert r.attempts == 3 and delays == [2.0, 2.0]


@pytest.mark.asyncio
async def test_retries_timeout_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json=ok_body())

    c, _ = make_client(handler)
    assert (await c.complete(MSGS)).attempts == 2


@pytest.mark.asyncio
async def test_non_retryable_fails_immediately():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(403, text="denied")

    c, delays = make_client(handler)
    with pytest.raises(LLMError) as e:
        await c.complete(MSGS)
    assert e.value.status == 403 and calls["n"] == 1 and delays == []


@pytest.mark.asyncio
async def test_exhausts_retries_on_500():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(500)

    c, _ = make_client(handler, max_retries=3)
    with pytest.raises(LLMError) as e:
        await c.complete(MSGS)
    assert e.value.status == 500 and calls["n"] == 4  # 1 + 3 retries


@pytest.mark.asyncio
async def test_concurrency_is_bounded():
    state = {"now": 0, "max": 0}

    async def handler(req):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return httpx.Response(200, json=ok_body())

    c, _ = make_client(handler, max_concurrency=3)
    results = await asyncio.gather(*[c.complete(MSGS) for _ in range(20)])
    assert len(results) == 20 and state["max"] == 3


@pytest.mark.asyncio
async def test_fault_injection_recovers_all_calls():
    rng = random.Random(0)

    def handler(req):
        if rng.random() < 0.3:
            return httpx.Response(rng.choice([429, 500, 503]))
        return httpx.Response(200, json=ok_body())

    c, _ = make_client(handler, max_concurrency=10, max_retries=10)
    results = await asyncio.gather(*[c.complete(MSGS) for _ in range(100)])
    assert all(r.text == "hello" for r in results)


@pytest.mark.asyncio
async def test_stream_yields_deltas():
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    c, _ = make_client(lambda r: httpx.Response(200, content=sse.encode()))
    parts = [d async for d in c.stream(MSGS)]
    assert parts == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_stream_retries_connect_errors():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

    c, _ = make_client(handler)
    assert [d async for d in c.stream(MSGS)] == ["ok"]


def test_count_tokens_positive_and_scales():
    c, _ = make_client(lambda r: httpx.Response(200, json=ok_body()))
    short, long = c.count_tokens("hello"), c.count_tokens("hello " * 100)
    assert 0 < short < long
    assert c.count_tokens(MSGS) > 0

@pytest.mark.asyncio
async def test_empty_content_with_length_warns(caplog):
    body = {"model": "m", "choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 50}}
    c, _ = make_client(lambda r: httpx.Response(200, json=body))
    with caplog.at_level("WARNING", logger="aegisflow.llm"):
        r = await c.complete(MSGS)
    assert r.text == "" and r.finish_reason == "length"
    assert "token budget exhausted" in caplog.text


@pytest.mark.asyncio
async def test_stream_ignores_reasoning_and_warns_when_no_content(caplog):
    sse = (
        'data: {"choices":[{"delta":{"reasoning":"thinking..."}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    c, _ = make_client(lambda r: httpx.Response(200, content=sse.encode()))
    with caplog.at_level("WARNING", logger="aegisflow.llm"):
        parts = [d async for d in c.stream(MSGS)]
    assert parts == [] and "token budget exhausted" in caplog.text


def test_prompt_overhead_applies_to_messages_only():
    c0, _ = make_client(lambda r: httpx.Response(200, json=ok_body()))
    c1, _ = make_client(lambda r: httpx.Response(200, json=ok_body()), prompt_overhead=67)
    assert c1.count_tokens(MSGS) == c0.count_tokens(MSGS) + 67
    assert c1.count_tokens("hello") == c0.count_tokens("hello")

def test_from_env_uses_measured_overhead(monkeypatch):
    from aegisflow.llm_gateway import from_env

    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("AEGIS_PROMPT_OVERHEAD", raising=False)
    monkeypatch.setenv("AEGIS_MODEL", "openai/gpt-oss-20b")
    assert from_env().prompt_overhead == 67
    monkeypatch.setenv("AEGIS_MODEL", "some/unmeasured-model")
    assert from_env().prompt_overhead == 0
    monkeypatch.setenv("AEGIS_PROMPT_OVERHEAD", "10")
    assert from_env().prompt_overhead == 10