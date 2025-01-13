"""Unit tests for disk cache backend."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from vllm.adapter_commons.storage.cache import (
    CacheConfig,
    DiskCacheConfig,
    CacheFactory,
)
from vllm.adapter_commons.storage.cache.base import (
    AdapterCacheError,
    CacheFullError,
    CacheKeyError,
    CacheIOError,
)
from vllm.adapter_commons.storage.cache.disk import DiskCache

# Test data
TEST_CONTENT = b"test content"
TEST_SIZE = len(TEST_CONTENT)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def cache_dir(temp_dir):
    """Create a directory for cache storage."""
    path = Path(temp_dir) / "cache"
    path.mkdir()
    return path


@pytest.fixture(scope="function")
def test_file(temp_dir) -> Path:
    """Create a test file with known content."""
    path = Path(temp_dir) / "test.bin"
    path.write_bytes(TEST_CONTENT)
    return path


@pytest.fixture(scope="function")
def disk_cache(cache_dir):
    """Create a disk cache instance."""
    config = DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=1024 * 1024,  # 1MB
        cache_dir=str(cache_dir),
        create_dirs=True,
    )
    return DiskCache(config)


@pytest.mark.asyncio
async def test_basic_operations(disk_cache, test_file, temp_dir):
    """Test basic cache operations."""
    key = "test_key"
    target_path = Path(temp_dir) / "cached.bin"
    
    # Test put
    await disk_cache.put(key, str(test_file))
    assert await disk_cache.contains(key)
    
    # Test get
    assert await disk_cache.get(key, str(target_path))
    assert target_path.read_bytes() == TEST_CONTENT
    
    # Test remove
    await disk_cache.remove(key)
    assert not await disk_cache.contains(key)


@pytest.mark.asyncio
async def test_size_tracking(disk_cache, test_file):
    """Test cache size tracking."""
    key = "test_key"
    initial_size = await disk_cache.get_current_size()
    
    # Add item
    await disk_cache.put(key, str(test_file))
    new_size = await disk_cache.get_current_size()
    assert new_size == initial_size + TEST_SIZE
    
    # Remove item
    await disk_cache.remove(key)
    final_size = await disk_cache.get_current_size()
    assert final_size == initial_size


@pytest.mark.asyncio
async def test_eviction_policy(disk_cache, temp_dir, cache_dir):
    """Test LRU eviction policy."""
    # Create small cache
    small_cache = DiskCache(DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=100,  # Very small
        cache_dir=str(cache_dir),
        create_dirs=True,
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
async def test_concurrent_access(disk_cache, test_file, temp_dir):
    """Test concurrent cache access."""
    async def cache_operation(index: int) -> None:
        key = f"key_{index}"
        target_path = Path(temp_dir) / f"concurrent_{index}.bin"
        
        # Put
        await disk_cache.put(key, str(test_file))
        assert await disk_cache.contains(key)
        
        # Get
        assert await disk_cache.get(key, str(target_path))
        assert target_path.read_bytes() == TEST_CONTENT
        
        # Remove
        await disk_cache.remove(key)
        assert not await disk_cache.contains(key)
    
    # Run concurrent operations
    tasks = [cache_operation(i) for i in range(5)]
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_error_handling(disk_cache, test_file, temp_dir):
    """Test error handling in cache operations."""
    key = "test_key"
    nonexistent_key = "nonexistent"
    target_path = Path(temp_dir) / "error_test.bin"
    
    # Test get with nonexistent key
    assert not await disk_cache.get(nonexistent_key, str(target_path))
    
    # Test remove nonexistent key
    await disk_cache.remove(nonexistent_key)  # Should not raise
    
    # Test cache full error
    small_cache = DiskCache(DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=10,  # Very small
        cache_dir=str(Path(temp_dir) / "small_cache"),
        create_dirs=True,
    ))
    
    with pytest.raises(CacheFullError):
        await small_cache.put(key, str(test_file))


@pytest.mark.asyncio
async def test_metadata(disk_cache, test_file):
    """Test metadata operations."""
    key = "test_key"
    
    # Put item and get metadata
    await disk_cache.put(key, str(test_file))
    metadata = await disk_cache.get_metadata(key)
    
    assert metadata is not None
    assert metadata["size"] == TEST_SIZE
    assert "last_access" in metadata
    assert "creation_time" in metadata


@pytest.mark.asyncio
async def test_clear(disk_cache, test_file):
    """Test cache clear operation."""
    # Add multiple items
    for i in range(5):
        await disk_cache.put(f"key_{i}", str(test_file))
    
    # Verify items exist
    for i in range(5):
        assert await disk_cache.contains(f"key_{i}")
    
    # Clear cache
    await disk_cache.clear()
    
    # Verify all items removed
    for i in range(5):
        assert not await disk_cache.contains(f"key_{i}")
    
    # Verify size is 0
    assert await disk_cache.get_current_size() == 0


@pytest.mark.asyncio
async def test_persistence(disk_cache, test_file, cache_dir):
    """Test cache persistence across instances."""
    key = "test_key"
    
    # Add item to first instance
    await disk_cache.put(key, str(test_file))
    
    # Create new instance with same cache dir
    new_cache = DiskCache(DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=1024 * 1024,
        cache_dir=str(cache_dir),
        create_dirs=True,
    ))
    
    # Verify item exists in new instance
    assert await new_cache.contains(key)
    metadata = await new_cache.get_metadata(key)
    assert metadata["size"] == TEST_SIZE


@pytest.mark.asyncio
async def test_file_permissions(disk_cache, test_file):
    """Test file permission handling."""
    key = "test_key"
    
    # Test with preserve_permissions=True
    cache_with_perms = DiskCache(DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=1024 * 1024,
        cache_dir=str(Path(disk_cache.config.cache_dir) / "with_perms"),
        create_dirs=True,
        preserve_permissions=True,
    ))
    
    await cache_with_perms.put(key, str(test_file))
    
    # Test with preserve_permissions=False
    cache_without_perms = DiskCache(DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=1024 * 1024,
        cache_dir=str(Path(disk_cache.config.cache_dir) / "without_perms"),
        create_dirs=True,
        preserve_permissions=False,
    ))
    
    await cache_without_perms.put(key, str(test_file))


@pytest.mark.asyncio
async def test_cleanup(disk_cache, test_file):
    """Test cleanup behavior."""
    key = "test_key"
    
    # Add item
    await disk_cache.put(key, str(test_file))
    
    # Cleanup
    await disk_cache.cleanup()
    
    # Verify cache is still usable
    assert await disk_cache.contains(key)
    
    # Cleanup after clear
    await disk_cache.clear()
    await disk_cache.cleanup()
    
    # Verify cache is still usable but empty
    assert not await disk_cache.contains(key)
    assert await disk_cache.get_current_size() == 0 