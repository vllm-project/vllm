# CPU platform configuration for CI testing
# This file forces vLLM to use CPU platform detection in CI environment

import os


def pytest_configure(config):
    """Force CPU platform before any test imports."""
    if os.getenv("CI"):
        # Set environment to force CPU platform detection
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Import and patch platform detection before tests run
        import vllm.platforms
        from vllm.platforms.cpu import CpuPlatform

        # Override current_platform to use CPU
        vllm.platforms._current_platform = CpuPlatform()
