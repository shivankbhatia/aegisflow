from typing import Any, Callable
import hashlib
import json


class GuardrailsInterceptor:

    def __init__(self, max_recursion: int = 3, max_tokens: int = 50000):
        self.max_recursion = max_recursion
        self.max_tokens = max_tokens
        self.tool_cache = {}

    def check_budget(self, total_tokens: int):
        if total_tokens > self.max_tokens:
            raise Exception(
                f"Budget Exceeded: Used {total_tokens} "
                f"tokens, limit is {self.max_tokens}"
            )

    def cache_tool_call(
        self,
        tool_name: str,
        kwargs: dict,
        execute_fn: Callable,
    ) -> Any:

        input_str = json.dumps(
            {
                "tool": tool_name,
                "kwargs": kwargs,
            },
            sort_keys=True,
        )

        call_hash = hashlib.sha256(
            input_str.encode()
        ).hexdigest()

        if call_hash in self.tool_cache:
            return self.tool_cache[call_hash]

        result = execute_fn(**kwargs)

        self.tool_cache[call_hash] = result

        return result

    def check_recursion(self, state: dict):
        count = state.get("recursion_count", 0)

        if count >= self.max_recursion:
            raise Exception(
                f"Maximum recursion limit of "
                f"{self.max_recursion} reached."
            )

        return count + 1