"""Test fixtures for cache backend tests."""

import os
import tempfile
from pathlib import Path

import pytest

# Test data
TEST_CONTENT = b"test content"
TEST_SIZE = len(TEST_CONTENT)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def test_file(temp_dir) -> Path:
    """Create a test file with known content."""
    path = Path(temp_dir) / "test.bin"
    path.write_bytes(TEST_CONTENT)
    return path


@pytest.fixture(scope="function")
def large_test_file(temp_dir) -> Path:
    """Create a large test file."""
    path = Path(temp_dir) / "large.bin"
    path.write_bytes(os.urandom(1024 * 1024))  # 1MB
    return path


@pytest.fixture(scope="function")
def cache_dir(temp_dir) -> Path:
    """Create a directory for cache storage."""
    path = Path(temp_dir) / "cache"
    path.mkdir()
    return path


@pytest.fixture(scope="function")
def metadata_file(temp_dir) -> Path:
    """Create a test metadata file."""
    path = Path(temp_dir) / "metadata.json"
    return path 