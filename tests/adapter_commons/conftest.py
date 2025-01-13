"""Common test fixtures and setup for adapter_commons tests."""

import os
import tempfile
from pathlib import Path

import pytest

from vllm.distributed import cleanup_dist_env_and_memory


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up the test environment."""
    # Set test environment variables
    os.environ["VLLM_USE_ASYNC_IO"] = "1"
    os.environ["VLLM_STORAGE_TEST_MODE"] = "1"
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup."""
    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    """Clean up after each test."""
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create and return a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        yield path 