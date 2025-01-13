"""Integration tests for adapter storage and cache components."""

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import boto3
import pytest
import pytest_asyncio
from moto import mock_aws

from vllm.adapter_commons.storage import (
    StorageConfig,
    S3StorageConfig,
    LocalStorageConfig,
    StorageProviderFactory,
)
from vllm.adapter_commons.storage.cache import (
    CacheConfig,
    MemoryCacheConfig,
    DiskCacheConfig,
    CacheFactory,
)

# Test data
TEST_ADAPTER_CONTENT = b"mock adapter content"
TEST_ADAPTER_SIZE = len(TEST_ADAPTER_CONTENT)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def local_adapter_path(temp_dir) -> Path:
    """Create a mock adapter file in the temp directory."""
    adapter_path = Path(temp_dir) / "test_adapter.bin"
    adapter_path.write_bytes(TEST_ADAPTER_CONTENT)
    yield adapter_path
    with suppress(FileNotFoundError):
        adapter_path.unlink()


@pytest.fixture(scope="function")
def mock_s3():
    """Create a mocked S3 environment."""
    with mock_aws():
        s3 = boto3.client("s3")
        # Create test bucket
        s3.create_bucket(Bucket="test-bucket")
        # Upload test adapter
        s3.put_object(
            Bucket="test-bucket",
            Key="adapters/test.bin",
            Body=TEST_ADAPTER_CONTENT
        )
        yield s3


@pytest.fixture(scope="function")
def storage_factory() -> StorageProviderFactory:
    """Create a storage provider factory instance."""
    return StorageProviderFactory()


@pytest.fixture(scope="function")
def cache_factory() -> CacheFactory:
    """Create a cache factory instance."""
    return CacheFactory()


@pytest.mark.asyncio
async def test_local_storage_download(
    temp_dir: str,
    local_adapter_path: Path,
    storage_factory: StorageProviderFactory,
):
    """Test local storage provider download functionality."""
    config = LocalStorageConfig(
        provider_type="local",
        allowed_paths=[temp_dir],
        create_dirs=True,
    )
    
    provider = storage_factory.create_provider(config)
    
    # Test download
    target_path = Path(temp_dir) / "downloaded_adapter.bin"
    await provider.download_adapter(
        str(local_adapter_path),
        str(target_path)
    )
    
    assert target_path.exists()
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT


@pytest.mark.asyncio
async def test_s3_storage_download(
    temp_dir: str,
    mock_s3,
    storage_factory: StorageProviderFactory,
):
    """Test S3 storage provider download functionality."""
    config = S3StorageConfig(
        provider_type="s3",
        region_name="us-east-1",
    )
    
    provider = storage_factory.create_provider(config)
    
    # Test download
    target_path = Path(temp_dir) / "downloaded_adapter.bin"
    await provider.download_adapter(
        "s3://test-bucket/adapters/test.bin",
        str(target_path)
    )
    
    assert target_path.exists()
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT


@pytest.mark.asyncio
async def test_memory_cache_operations(
    temp_dir: str,
    local_adapter_path: Path,
    cache_factory: CacheFactory,
):
    """Test memory cache backend operations."""
    config = MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=1024 * 1024,  # 1MB
    )
    
    cache = cache_factory.create_backend(config)
    
    # Test put
    key = "test_adapter"
    await cache.put(key, str(local_adapter_path))
    
    # Test get
    target_path = Path(temp_dir) / "cached_adapter.bin"
    assert await cache.get(key, str(target_path))
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT
    
    # Test remove
    await cache.remove(key)
    assert not await cache.contains(key)


@pytest.mark.asyncio
async def test_disk_cache_operations(
    temp_dir: str,
    local_adapter_path: Path,
    cache_factory: CacheFactory,
):
    """Test disk cache backend operations."""
    cache_dir = Path(temp_dir) / "cache"
    config = DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=1024 * 1024,  # 1MB
        cache_dir=str(cache_dir),
    )
    
    cache = cache_factory.create_backend(config)
    
    # Test put
    key = "test_adapter"
    await cache.put(key, str(local_adapter_path))
    
    # Test get
    target_path = Path(temp_dir) / "cached_adapter.bin"
    assert await cache.get(key, str(target_path))
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT
    
    # Test remove
    await cache.remove(key)
    assert not await cache.contains(key)


@pytest.mark.asyncio
async def test_storage_with_cache_integration(
    temp_dir: str,
    mock_s3,
    storage_factory: StorageProviderFactory,
    cache_factory: CacheFactory,
):
    """Test integration between storage provider and cache backend."""
    # Configure storage
    storage_config = S3StorageConfig(
        provider_type="s3",
        region_name="us-east-1",
    )
    storage = storage_factory.create_provider(storage_config)
    
    # Configure cache
    cache_config = MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=1024 * 1024,  # 1MB
    )
    cache = cache_factory.create_backend(cache_config)
    
    # Test download and cache flow
    s3_uri = "s3://test-bucket/adapters/test.bin"
    cache_key = "test_adapter"
    
    # First download (should go to S3)
    target_path = Path(temp_dir) / "first_download.bin"
    await storage.download_adapter(s3_uri, str(target_path))
    assert target_path.read_bytes() == TEST_ADAPTER_CONTENT
    
    # Cache the adapter
    await cache.put(cache_key, str(target_path))
    
    # Second download (should come from cache)
    cached_path = Path(temp_dir) / "cached_download.bin"
    assert await cache.get(cache_key, str(cached_path))
    assert cached_path.read_bytes() == TEST_ADAPTER_CONTENT


@pytest.mark.asyncio
async def test_error_handling(
    temp_dir: str,
    mock_s3,
    storage_factory: StorageProviderFactory,
    cache_factory: CacheFactory,
):
    """Test error handling in storage and cache operations."""
    # Storage tests
    storage_config = S3StorageConfig(
        provider_type="s3",
        region_name="us-east-1",
    )
    storage = storage_factory.create_provider(storage_config)
    
    # Test invalid S3 URI
    with pytest.raises(Exception):
        await storage.download_adapter(
            "s3://nonexistent-bucket/fake.bin",
            str(Path(temp_dir) / "should_not_exist.bin")
        )
    
    # Cache tests
    cache_config = MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=10,  # Very small to force errors
    )
    cache = cache_factory.create_backend(cache_config)
    
    # Test cache full error
    with pytest.raises(Exception):
        await cache.put(
            "test_key",
            str(Path(temp_dir) / "large_file.bin")
        ) 