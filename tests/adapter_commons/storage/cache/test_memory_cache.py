"""Unit tests for memory cache backend."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from vllm.adapter_commons.storage.cache import (
    CacheConfig,
    MemoryCacheConfig,
    CacheFactory,
)
from vllm.adapter_commons.storage.cache.base import (
    AdapterCacheError,
    CacheFullError,
    CacheKeyError,
    CacheIOError,
)
from vllm.adapter_commons.storage.cache.memory import MemoryCache

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
def memory_cache():
    """Create a memory cache instance."""
    config = MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=1024 * 1024,  # 1MB
        max_items=100,
    )
    return MemoryCache(config)


@pytest.mark.asyncio
async def test_basic_operations(memory_cache, test_file, temp_dir):
    """Test basic cache operations."""
    key = "test_key"
    target_path = Path(temp_dir) / "cached.bin"
    
    # Test put
    await memory_cache.put(key, str(test_file))
    assert await memory_cache.contains(key)
    
    # Test get
    assert await memory_cache.get(key, str(target_path))
    assert target_path.read_bytes() == TEST_CONTENT
    
    # Test remove
    await memory_cache.remove(key)
    assert not await memory_cache.contains(key)


@pytest.mark.asyncio
async def test_size_tracking(memory_cache, test_file):
    """Test cache size tracking."""
    key = "test_key"
    initial_size = await memory_cache.get_current_size()
    
    # Add item
    await memory_cache.put(key, str(test_file))
    new_size = await memory_cache.get_current_size()
    assert new_size == initial_size + TEST_SIZE
    
    # Remove item
    await memory_cache.remove(key)
    final_size = await memory_cache.get_current_size()
    assert final_size == initial_size


@pytest.mark.asyncio
async def test_eviction_policy(memory_cache, temp_dir):
    """Test LRU eviction policy."""
    # Create small cache
    small_cache = MemoryCache(MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=100,  # Very small
        max_items=2,
    ))
    
    # Create test files
    files = []
    for i in range(3):
        path = Path(temp_dir) / f"test_{i}.bin"
        path.write_bytes(b"x" * 40)  # Each file is 40 bytes
        files.append(path)
    
    # Add files to cache
    for i, file in enumerate(files):
        await small_cache.put(f"key_{i}", str(file))
        await asyncio.sleep(0.1)  # Ensure different access times
    
    # Verify oldest item was evicted
    assert not await small_cache.contains("key_0")
    assert await small_cache.contains("key_1")
    assert await small_cache.contains("key_2")


@pytest.mark.asyncio
async def test_concurrent_access(memory_cache, test_file, temp_dir):
    """Test concurrent cache access."""
    async def cache_operation(index: int) -> None:
        key = f"key_{index}"
        target_path = Path(temp_dir) / f"concurrent_{index}.bin"
        
        # Put
        await memory_cache.put(key, str(test_file))
        assert await memory_cache.contains(key)
        
        # Get
        assert await memory_cache.get(key, str(target_path))
        assert target_path.read_bytes() == TEST_CONTENT
        
        # Remove
        await memory_cache.remove(key)
        assert not await memory_cache.contains(key)
    
    # Run concurrent operations
    tasks = [cache_operation(i) for i in range(5)]
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_error_handling(memory_cache, test_file, temp_dir):
    """Test error handling in cache operations."""
    key = "test_key"
    nonexistent_key = "nonexistent"
    target_path = Path(temp_dir) / "error_test.bin"
    
    # Test get with nonexistent key
    assert not await memory_cache.get(nonexistent_key, str(target_path))
    
    # Test remove nonexistent key
    await memory_cache.remove(nonexistent_key)  # Should not raise
    
    # Test cache full error
    small_cache = MemoryCache(MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=10,  # Very small
        max_items=1,
    ))
    
    with pytest.raises(CacheFullError):
        await small_cache.put(key, str(test_file))


@pytest.mark.asyncio
async def test_metadata(memory_cache, test_file):
    """Test metadata operations."""
    key = "test_key"
    
    # Put item and get metadata
    await memory_cache.put(key, str(test_file))
    metadata = await memory_cache.get_metadata(key)
    
    assert metadata is not None
    assert metadata["size"] == TEST_SIZE
    assert "last_access" in metadata
    assert "creation_time" in metadata


@pytest.mark.asyncio
async def test_clear(memory_cache, test_file):
    """Test cache clear operation."""
    # Add multiple items
    for i in range(5):
        await memory_cache.put(f"key_{i}", str(test_file))
    
    # Verify items exist
    for i in range(5):
        assert await memory_cache.contains(f"key_{i}")
    
    # Clear cache
    await memory_cache.clear()
    
    # Verify all items removed
    for i in range(5):
        assert not await memory_cache.contains(f"key_{i}")
    
    # Verify size is 0
    assert await memory_cache.get_current_size() == 0


@pytest.mark.asyncio
async def test_max_items_limit(memory_cache, test_file):
    """Test maximum items limit."""
    small_cache = MemoryCache(MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=1024 * 1024,  # Large enough
        max_items=2,  # Small item limit
    ))
    
    # Add items up to limit
    await small_cache.put("key_1", str(test_file))
    await small_cache.put("key_2", str(test_file))
    
    # Try to add one more
    with pytest.raises(CacheFullError):
        await small_cache.put("key_3", str(test_file))


@pytest.mark.asyncio
async def test_cleanup(memory_cache, test_file):
    """Test cleanup behavior."""
    key = "test_key"
    
    # Add item
    await memory_cache.put(key, str(test_file))
    
    # Cleanup
    await memory_cache.cleanup()
    
    # Verify cache is still usable
    assert await memory_cache.contains(key)
    
    # Cleanup after clear
    await memory_cache.clear()
    await memory_cache.cleanup()
    
    # Verify cache is still usable but empty
    assert not await memory_cache.contains(key)
    assert await memory_cache.get_current_size() == 0 