import re

from ..llm.client import get_llm
from .base import CoderOutputSchema

_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```\s*$")


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


class CoderAgent:

    def __init__(self, llm_model: str | None = None):
        self.llm = get_llm(llm_model)

    def process(
        self,
        plan: str,
        files: list[str],
        feedback: str = "",
        previous_code: str = "",
        previous_tests: str = "",
    ) -> CoderOutputSchema:

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

        structured_llm = self.llm.with_structured_output(
            CoderOutputSchema
        )

        result = structured_llm.invoke(prompt)
        result.code_patch = _strip_fences(result.code_patch)
        result.test_patch = _strip_fences(result.test_patch)
        return result