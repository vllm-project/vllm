"""Unit tests for cache factory."""

import os
import pytest
from pathlib import Path
from typing import Type
from unittest.mock import Mock

from vllm.adapter_commons.storage.cache import (
    CacheFactory,
    AdapterCacheBackend,
    MemoryCache,
    DiskCache,
    CacheConfig,
    MemoryCacheConfig,
    DiskCacheConfig,
)

# Test data
TEST_BACKEND_TYPE = "test_backend"


class MockCacheBackend(AdapterCacheBackend):
    """Mock cache backend for testing."""
    async def get(self, key: str, target_path: str) -> bool:
        return True
    
    async def put(self, key: str, source_path: str) -> None:
        pass
    
    async def remove(self, key: str) -> None:
        pass
    
    async def clear(self) -> None:
        pass
    
    async def contains(self, key: str) -> bool:
        return True
    
    async def get_size(self, key: str) -> int:
        return 0
    
    async def get_metadata(self, key: str) -> dict:
        return {}
    
    async def iter_keys(self) -> list:
        return []
    
    async def cleanup(self) -> None:
        pass
    
    @property
    def max_size(self) -> int:
        return 1024
    
    @property
    def current_size(self) -> int:
        return 0


@pytest.fixture(scope="function")
def mock_backend():
    """Create a mock cache backend class."""
    return MockCacheBackend


@pytest.fixture(scope="function")
def factory():
    """Create a fresh factory instance for each test."""
    # Reset class registry
    CacheFactory._backend_registry = {
        "memory": MemoryCache,
        "disk": DiskCache,
    }
    return CacheFactory()


def test_default_backends(factory):
    """Test default backend registration."""
    backends = factory.get_registered_backends()
    assert "memory" in backends
    assert "disk" in backends
    assert len(backends) == 2


def test_register_backend(factory, mock_backend):
    """Test backend registration."""
    # Register new backend
    CacheFactory.register_backend(
        TEST_BACKEND_TYPE,
        mock_backend
    )
    
    # Verify registration
    backends = factory.get_registered_backends()
    assert TEST_BACKEND_TYPE in backends


def test_register_invalid_backend(factory):
    """Test registration with invalid backend class."""
    class InvalidBackend:
        pass
    
    with pytest.raises(ValueError):
        CacheFactory.register_backend(
            TEST_BACKEND_TYPE,
            InvalidBackend
        )


def test_register_duplicate_backend(factory, mock_backend):
    """Test registration with duplicate backend type."""
    CacheFactory.register_backend(
        TEST_BACKEND_TYPE,
        mock_backend
    )
    
    with pytest.raises(ValueError):
        CacheFactory.register_backend(
            TEST_BACKEND_TYPE,
            mock_backend
        )


def test_create_backend_memory(factory):
    """Test creating memory cache backend."""
    config = MemoryCacheConfig(
        backend_type="memory",
        max_size_bytes=1024 * 1024
    )
    
    backend = factory.create_backend(config)
    assert isinstance(backend, MemoryCache)


def test_create_backend_disk(factory, tmp_path):
    """Test creating disk cache backend."""
    config = DiskCacheConfig(
        backend_type="disk",
        max_size_bytes=1024 * 1024,
        cache_dir=str(tmp_path)
    )
    
    backend = factory.create_backend(config)
    assert isinstance(backend, DiskCache)


def test_create_backend_unknown_type(factory):
    """Test creating backend with unknown type."""
    config = CacheConfig(backend_type="unknown")
    
    with pytest.raises(ValueError):
        factory.create_backend(config)


def test_create_backend_mismatched_config(factory):
    """Test creating backend with mismatched config type."""
    # Try to create memory backend with disk config
    config = DiskCacheConfig(
        backend_type="memory",
        max_size_bytes=1024 * 1024,
        cache_dir="/tmp/cache"
    )
    
    with pytest.raises(ValueError):
        factory.create_backend(config)


def test_create_backend_with_custom(factory, mock_backend):
    """Test creating custom backend."""
    # Register custom backend
    CacheFactory.register_backend(
        TEST_BACKEND_TYPE,
        mock_backend
    )
    
    # Create config for custom backend
    config = CacheConfig(
        backend_type=TEST_BACKEND_TYPE,
        max_size_bytes=1024 * 1024
    )
    
    backend = factory.create_backend(config)
    assert isinstance(backend, mock_backend) 