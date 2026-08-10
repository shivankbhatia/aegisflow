from typing import Any, Callable
import hashlib
import json

class GuardrailsInterceptor:
    def __init__(self, max_recursion: int = 3, max_tokens: int = 10000):
        self.max_recursion = max_recursion
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.tool_cache = {}

    def track_tokens(self, tokens_used: int):
        self.current_tokens += tokens_used
        if self.current_tokens > self.max_tokens:
            raise Exception(f"Budget Exceeded: Used {self.current_tokens} tokens, limit is {self.max_tokens}")

    def cache_tool_call(self, tool_name: str, kwargs: dict, execute_fn: Callable) -> Any:
        # Create deterministic hash of inputs
        input_str = json.dumps({"tool": tool_name, "kwargs": kwargs}, sort_keys=True)
        call_hash = hashlib.sha256(input_str.encode()).hexdigest()

        if call_hash in self.tool_cache:
            return self.tool_cache[call_hash]

        result = execute_fn(**kwargs)
        self.tool_cache[call_hash] = result
        return result

    def check_recursion(self, state: dict):
        # A simple check: if the last N actions are identical, break loop
        messages = state.get("messages", [])
        if len(messages) >= self.max_recursion * 2:
            # Check for cyclical repeating patterns (simplified)
            recent_tool_calls = [m.tool_calls for m in messages[-4:] if hasattr(m, 'tool_calls') and m.tool_calls]
            if len(recent_tool_calls) >= 2 and recent_tool_calls[-1] == recent_tool_calls[-2]:
                raise Exception("Recursion Limit Exceeded: Agent is stuck in a loop calling the same tools.")
