from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage
    latency_s: float
    attempts: int  # 1 = succeeded first try
    finish_reason: str | None = None  # "stop" | "length" | ...


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status