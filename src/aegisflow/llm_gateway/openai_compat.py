import asyncio
import json
import logging
import math
import os
import random
import time
from typing import AsyncIterator, Awaitable, Callable, Sequence

import httpx

from .types import LLMError, LLMResponse, Message, Usage

log = logging.getLogger("aegisflow.llm")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
# Fixed per-request prompt tokens added by the provider's chat template.
# Measured with scripts/gateway_smoke.py; add models as you measure them.
PROMPT_OVERHEAD = {"openai/gpt-oss-20b": 67}

_encoder = None


def _count(text: str) -> int:
    """tiktoken o200k_base is an approximation for non-OpenAI models.
    Falls back to ~4 chars/token if the encoding can't be loaded."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("o200k_base")
        except Exception:
            _encoder = False
    if _encoder:
        return len(_encoder.encode(text))
    return math.ceil(len(text) / 4)


class OpenAICompatClient:
    """Async client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_concurrency: int = 8,
        max_retries: int = 4,
        timeout: float = 60.0,
        backoff_base: float = 0.5,
        backoff_cap: float = 20.0,
        prompt_overhead: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        # Fixed tokens the provider adds per request (chat template / system
        # preamble). Measure with scripts/gateway_smoke.py.
        self.prompt_overhead = prompt_overhead
        self._sleep = sleep
        self._sem = asyncio.Semaphore(max_concurrency)
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    # ---- internals -------------------------------------------------------

    def _payload(self, messages: Sequence[Message], **opts) -> dict:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            **opts,
        }

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_cap)
        exp = min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))
        return exp * (0.5 + random.random() * 0.5)  # jitter

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        try:
            return float(resp.headers["retry-after"])
        except (KeyError, ValueError):
            return None

    async def _request(self, payload: dict) -> tuple[httpx.Response, int]:
        attempts = 0
        while True:
            attempts += 1
            retry_after = None
            try:
                async with self._sem:  # released during backoff sleep
                    resp = await self._http.post("/chat/completions", json=payload)
                if resp.status_code < 400:
                    return resp, attempts
                err = LLMError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code
                )
                retryable = resp.status_code in RETRYABLE_STATUS
                retry_after = self._retry_after(resp)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                err, retryable = LLMError(f"{type(e).__name__}: {e}"), True

            if not retryable or attempts > self.max_retries:
                raise err
            delay = self._delay(attempts, retry_after)
            log.warning("llm_retry attempt=%d delay=%.2fs reason=%s", attempts, delay, err)
            await self._sleep(delay)

    # ---- public API ------------------------------------------------------

    async def complete(self, messages: Sequence[Message], **opts) -> LLMResponse:
        t0 = time.perf_counter()
        resp, attempts = await self._request(self._payload(messages, **opts))
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
        u = data.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        latency = time.perf_counter() - t0
        log.info(
            "llm_call model=%s prompt_tokens=%d completion_tokens=%d latency_s=%.3f attempts=%d",
            self.model, usage.prompt_tokens, usage.completion_tokens, latency, attempts,
        )
        if not text and finish == "length":
            log.warning(
                "empty content, token budget exhausted (reasoning model?): "
                "raise max_tokens or lower reasoning_effort"
            )
        return LLMResponse(
            text, data.get("model", self.model), usage, latency, attempts, finish
        )

    async def stream(self, messages: Sequence[Message], **opts) -> AsyncIterator[str]:
        """Yields text deltas. Retries only while connecting; a stream that
        fails mid-way raises. Holds a concurrency slot for its whole life."""
        payload = self._payload(messages, stream=True, **opts)
        async with self._sem:
            attempts = 0
            while True:
                attempts += 1
                retry_after = None
                try:
                    req = self._http.build_request("POST", "/chat/completions", json=payload)
                    resp = await self._http.send(req, stream=True)
                    if resp.status_code < 400:
                        break
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    await resp.aclose()
                    err = LLMError(f"HTTP {resp.status_code}: {body}", resp.status_code)
                    retryable = resp.status_code in RETRYABLE_STATUS
                    retry_after = self._retry_after(resp)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    err, retryable = LLMError(f"{type(e).__name__}: {e}"), True
                if not retryable or attempts > self.max_retries:
                    raise err
                await self._sleep(self._delay(attempts, retry_after))

            got_content, finish = False, None
            try:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    choices = json.loads(chunk).get("choices") or []
                    if not choices:
                        continue
                    # delta.reasoning (reasoning models) is intentionally ignored
                    delta = choices[0].get("delta", {}).get("content")
                    finish = choices[0].get("finish_reason") or finish
                    if delta:
                        got_content = True
                        yield delta
            finally:
                await resp.aclose()
            if not got_content and finish == "length":
                log.warning(
                    "stream produced no content, token budget exhausted "
                    "(reasoning model?): raise max_tokens or lower reasoning_effort"
                )

    def count_tokens(self, content: "str | Sequence[Message]") -> int:
        if isinstance(content, str):
            return _count(content)
        framing = sum(_count(m.content) + 4 for m in content)  # ~4 tokens/msg
        return framing + self.prompt_overhead

    async def aclose(self) -> None:
        await self._http.aclose()


def from_env(**overrides) -> OpenAICompatClient:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    model = os.getenv("AEGIS_MODEL", DEFAULT_MODEL)
    overrides.setdefault(
        "prompt_overhead",
        int(os.getenv("AEGIS_PROMPT_OVERHEAD", PROMPT_OVERHEAD.get(model, 0))),
    )
    return OpenAICompatClient(
        base_url=os.getenv("AEGIS_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        model=model,
        **overrides,
    )