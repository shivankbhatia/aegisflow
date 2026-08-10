import docker
from typing import Tuple

class SandboxExecutor:
    def __init__(self, image: str = "python:3.11-slim"):
        self.client = docker.from_env()
        self.image = image
        # Ensure image is pulled (this might take a bit on first run)
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            self.client.images.pull(self.image)

    def execute_code(self, code: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Executes code in a secure Docker sandbox.
        Returns a tuple of (success_boolean, output_string)
        """
        # We will spin up an ephemeral container, run the code, and capture output.
        try:
            # We use a simple python -c command
            container = self.client.containers.run(
                self.image,
                command=["python", "-c", code],
                detach=True,
                mem_limit="128m",
                network_disabled=True, # Prevent external network access
            )
            
            # Wait for execution with timeout
            result = container.wait(timeout=timeout)
            
            # Get logs
            logs = container.logs().decode("utf-8")
            
            container.remove(force=True)
            
            success = result.get("StatusCode", 1) == 0
            return success, logs
            
        except Exception as e:
            return False, f"Sandbox Error: {str(e)}"
