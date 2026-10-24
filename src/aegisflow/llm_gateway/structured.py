"""Schema-validated LLM output: constrained decoding + validate/repair loop."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Generic, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .base import LLMClient
from .types import LLMError, Message, Usage

log = logging.getLogger("aegisflow.llm")
T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class StructuredOutputError(Exception):
    def __init__(self, message: str, attempts: int, last_raw: str, usage: Usage | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw
        self.usage = usage or Usage()


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    attempts: int  # validation attempts; 1 = valid on first try
    usage: Usage  # summed across all attempts
    raw: str


def extract_json(text: str) -> str:
    """Tolerate markdown fences and prose around a single JSON object."""
    t = _FENCE.sub("", text.strip()).strip()
    if t.startswith("{") or t.startswith("["):
        return t
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if 0 <= start < end else t


def _format_errors(e: ValidationError, limit: int = 8) -> str:
    lines = [
        f"- {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
        for err in e.errors()[:limit]
    ]
    return "\n".join(lines)


def _prompt_mode_messages(messages: Sequence[Message], schema: Type[BaseModel]) -> list[Message]:
    instr = (
        "Respond with ONLY a JSON object matching this JSON Schema. "
        "No prose, no markdown fences.\n" + json.dumps(schema.model_json_schema())
    )
    return [Message("system", instr), *messages]


async def complete_structured(
    client: LLMClient,
    messages: Sequence[Message],
    schema: Type[T],
    *,
    max_repairs: int = 2,
    mode: str = "json_schema",  # "json_schema" | "prompt"
    fallback: bool = True,  # json_schema -> prompt if the provider returns 400
    **opts,
) -> StructuredResult[T]:
    opts = dict(opts)
    msgs = list(messages)
    if mode == "json_schema":
        opts["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
    else:
        msgs = _prompt_mode_messages(msgs, schema)

    total = Usage()
    last_raw = ""
    for attempt in range(1, max_repairs + 2):
        try:
            resp = await client.complete(msgs, **opts)
        except LLMError as e:
            if mode == "json_schema" and fallback and e.status == 400 and attempt == 1:
                log.warning("response_format rejected (%s); falling back to prompt mode", e)
                mode = "prompt"
                opts.pop("response_format", None)
                msgs = _prompt_mode_messages(messages, schema)
                continue
            raise

        total = Usage(
            total.prompt_tokens + resp.usage.prompt_tokens,
            total.completion_tokens + resp.usage.completion_tokens,
        )
        last_raw = resp.text

        if resp.finish_reason == "length":
            raise StructuredOutputError(
                "output truncated (finish_reason=length): raise max_tokens "
                "or lower reasoning_effort; repairing cannot fix this",
                attempt, last_raw, total,
            )
        try:
            value = schema.model_validate_json(extract_json(resp.text))
            return StructuredResult(value, attempt, total, resp.text)
        except ValidationError as e:
            problem = _format_errors(e)
            log.warning("structured_invalid attempt=%d schema=%s errors=%s", attempt, schema.__name__, problem)
            msgs = [
                *msgs,
                Message("assistant", resp.text or "(empty)"),
                Message("user", f"Your reply failed validation:\n{problem}\nReturn the corrected JSON object only."),
            ]

    raise StructuredOutputError(
        f"no valid {schema.__name__} after {max_repairs + 1} attempts",
        max_repairs + 1, last_raw, total,
    )
