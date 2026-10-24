import os

import pytest

from aegisflow.orchestrator.workflow import run_workflow

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY and Docker"
)


@pytest.mark.asyncio
async def test_workflow_live():
    result = await run_workflow("Fix the hello function so it returns 'Hello, World!'")
    assert "approved" in result