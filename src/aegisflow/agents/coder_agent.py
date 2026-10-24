import re

from ..llm_gateway import LLMClient, Message, StructuredResult, complete_structured
from .base import LLM_OPTS, CoderOutputSchema

_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```\s*$")


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


class CoderAgent:

    def __init__(self, client: LLMClient):
        self.client = client

    async def process(
        self,
        plan: str,
        files: list[str],
        feedback: str = "",
        previous_code: str = "",
        previous_tests: str = "",
    ) -> StructuredResult[CoderOutputSchema]:

        retry_block = ""
        if feedback:
            retry_block = f"""
Your previous attempt was rejected.

Reviewer feedback:
{feedback}

Previous code:
{previous_code}

Previous tests:
{previous_tests}

Fix the problems above.
"""

        prompt = f"""
You are a Coder Agent.

Implement the fix for the plan below.

Rules:
- code_patch must be complete, self-contained, runnable Python source.
  Do NOT output a diff or patch format. Do NOT use markdown fences.
- test_patch must be plain `assert` statements that call the code defined
  in code_patch. Do not import from files, do not use pytest.
- Code and tests are concatenated and run as one script in an empty
  sandbox with no network, so nothing may depend on other files.

Plan:
{plan}

Relevant Files:
{', '.join(files)}
{retry_block}"""

        res = await complete_structured(
            self.client, [Message("user", prompt)], CoderOutputSchema, **LLM_OPTS
        )
        res.value.code_patch = _strip_fences(res.value.code_patch)
        res.value.test_patch = _strip_fences(res.value.test_patch)
        return res