from aegisflow.orchestrator.workflow import run_workflow


def test_workflow():

    result = run_workflow(
        "Fix the hello function so it returns 'Hello, World!'"
    )

    print(result)

    assert "approved" in result