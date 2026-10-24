from typing import AsyncIterator, Protocol, Sequence, runtime_checkable

from .types import LLMResponse, Message


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic interface. Agents depend on this, never on an SDK."""

    async def complete(self, messages: Sequence[Message], **opts) -> LLMResponse: ...

    def stream(self, messages: Sequence[Message], **opts) -> AsyncIterator[str]: ...

    def count_tokens(self, content: "str | Sequence[Message]") -> int: ...

    async def aclose(self) -> None: ...