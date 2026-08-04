# SPDX-License-Identifier: Apache-2.0
"""
Test cases for  storage_plugin_launcher.

This module tests the storage plugins loading mechanism in CreateStorageBackends.
It creates a simple mock storage plugin extends StoragePluginInterface
and verifies that:
1. The backend is properly loaded via the configuration
2. The backend methods can be called through StorageManager
"""

# Standard
from typing import Any, Callable, List, Optional, Sequence
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from tests.v1.utils import create_test_memory_obj


class MockStoragePlugin(StoragePluginInterface):
    """
    A mock storage plugin for testing the storage plugin functionality.
    This storage plugin extends StoragePluginInterface and provides simple
    in-memory storage for testing purposes.

    This plugin uses instance-level storage for call history and data storage
    to enable proper testing of dynamically loaded storage plugins.
    """

    def __init__(
        self,
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop=None,
        dst_device: str = "cpu",
    ):
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        self._allocator = AdHocMemoryAllocator(device="cpu")
        # Use instance-level storage
        self.call_history: List[str] = []
        self.storage: dict = {}

    def __str__(self):
        return "MockStoragePlugin"

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in storage."""
        self.call_history.append(f"contains:{key.chunk_hash}")
        return key.chunk_hash in self.storage

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check if key is in ongoing put tasks."""
        self.call_history.append(f"exists_in_put_tasks:{key.chunk_hash}")
        return False

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Submit a batched put task."""
        for key, obj in zip(keys, objs, strict=True):
            self.call_history.append(f"batched_submit_put_task:{key.chunk_hash}")
            self.storage[key.chunk_hash] = obj

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking get for a single key."""
        self.call_history.append(f"get_blocking:{key.chunk_hash}")
        return self.storage.get(key.chunk_hash)

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a key in storage."""
        self.call_history.append(f"pin:{key.chunk_hash}")
        return key.chunk_hash in self.storage

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a key in storage."""
        self.call_history.append(f"unpin:{key.chunk_hash}")
        return key.chunk_hash in self.storage

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a key from storage."""
        self.call_history.append(f"remove:{key.chunk_hash}")
        if key.chunk_hash in self.storage:
            del self.storage[key.chunk_hash]
            return True
        return False

    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        """Get the allocator backend."""
        # Return the LocalCPUBackend if available, otherwise use self
        if self.local_cpu_backend is not None:
            return self.local_cpu_backend
        raise RuntimeError("No allocator backend available")

    def close(self) -> None:
        """Close the backend."""
        self.call_history.append("close")


# Module-level constants for storage plugin configuration
MOCK_BACKEND_EXTRA_CONFIG = {
    "storage_plugin.mock_backend.module_path": (
        "tests.v1.storage_backend.test_storage_plugin"
    ),
    "storage_plugin.mock_backend.class_name": "MockStoragePlugin",
}
MOCK_BACKEND_STORAGE_PLUGINS = ["mock_backend"]


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


def create_test_config(
    extra_config: Optional[dict] = None,
    storage_plugins: Optional[List[str]] = None,
):
    """Create a test configuration for storage plugin testing."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,  # Small size for testing
        lmcache_instance_id="test_storage_plugin_instance",
    )
    if extra_config:
        config.extra_config = extra_config
    if storage_plugins:
        config.storage_plugins = storage_plugins
    return config


def create_test_metadata():
    """Create test metadata for testing."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            4,
            2,
            256,
            8,
            128,
        ),  # (num_layers, 2, chunk_size, num_heads, head_dim)
    )


def get_mock_backend(storage_manager_or_backends):
    """
    Get the mock storage plugin instance from storage manager or backends dict.
    Handles both StorageManager and OrderedDict[str, StorageBackendInterface].
    """
    if hasattr(storage_manager_or_backends, "storage_backends"):
        backends = storage_manager_or_backends.storage_backends
    else:
        backends = storage_manager_or_backends

    return backends.get("mock_backend")


class TestCreateDynamicBackends:
    """Test cases for storage_plugin_launcher functionality."""

    @pytest.fixture
    def event_manager(self):
        """Create an EventManager for testing."""
        return EventManager()

    @pytest.fixture
    def async_loop(self):
        """Create an async event loop for testing."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def storage_manager_with_mock_backend(self, event_manager):
        """Fixture to create a StorageManager with the mock storage plugin loaded."""
        config = create_test_config(
            extra_config=MOCK_BACKEND_EXTRA_CONFIG,
            storage_plugins=MOCK_BACKEND_STORAGE_PLUGINS,
        )
        metadata = create_test_metadata()

        storage_manager = StorageManager(
            config=config,
            metadata=metadata,
            event_manager=event_manager,
        )

        yield storage_manager

        storage_manager.close()

    def test_storage_plugin_is_loaded(self, async_loop):
        """
        Test that a storage plugin is properly loaded via CreateStorageBackends.
        """
        config = create_test_config(
            extra_config=MOCK_BACKEND_EXTRA_CONFIG,
            storage_plugins=MOCK_BACKEND_STORAGE_PLUGINS,
        )
        metadata = create_test_metadata()

        # Create storage backends
        storage_backends = CreateStorageBackends(
            config=config,
            metadata=metadata,
            loop=async_loop,
            dst_device="cpu",
        )

        # Verify the dynamic backend is loaded
        assert "mock_backend" in storage_backends, (
            f"Expected 'mock_backend' in storage_backends, "
            f"got: {list(storage_backends.keys())}"
        )

        # Verify it's an instance by checking class name
        # (since the dynamically imported class is a different object)
        mock_backend = storage_backends["mock_backend"]
        assert mock_backend.__class__.__name__ == MockStoragePlugin.__name__, (
            f"Expected MockStoragePlugin instance, got: {type(mock_backend)}"
        )

        # Verify it extends StoragePluginInterface
        assert isinstance(mock_backend, StoragePluginInterface), (
            f"Expected StoragePluginInterface instance, got: {type(mock_backend)}"
        )

        # Verify the backend has the expected attributes (call_history, storage)
        assert hasattr(mock_backend, "call_history"), (
            "Expected mock_backend to have 'call_history' attribute"
        )
        assert hasattr(mock_backend, "storage"), (
            "Expected mock_backend to have 'storage' attribute"
        )

        # Close all backends
        for backend in storage_backends.values():
            backend.close()

    def test_dynamic_backend_contains_called_via_storage_manager(
        self, storage_manager_with_mock_backend
    ):
        """
        Test that StorageManager calls the dynamic backend's contains method.
        """
        storage_manager = storage_manager_with_mock_backend

        # Verify MockStoragePlugin is in the storage backends
        assert "mock_backend" in storage_manager.storage_backends, (
            f"Expected 'mock_backend' in storage_backends, "
            f"got: {list(storage_manager.storage_backends.keys())}"
        )

        # Get the mock backend instance to access its call history
        mock_backend = get_mock_backend(storage_manager)
        assert mock_backend is not None, "Expected mock_backend to be loaded"

        # Clear the call history before the test
        mock_backend.call_history.clear()

        # Create a test key
        test_key = create_test_key(key_id=12345)

        # Call contains via storage_manager
        # The contains method iterates over all backends
        result = storage_manager.contains(test_key)

        # Verify the dynamic backend's contains was called
        assert any("contains:12345" in call for call in mock_backend.call_history), (
            f"Expected 'contains:12345' in call history, "
            f"got: {mock_backend.call_history}"
        )

        # The key doesn't exist, so result should be None
        assert result is None, f"Expected None, got: {result}"

    def test_dynamic_backend_get_blocking_called_via_storage_manager(
        self, storage_manager_with_mock_backend
    ):
        """
        Test that StorageManager can retrieve data through the dynamic backend.
        """
        storage_manager = storage_manager_with_mock_backend

        # Get the mock backend instance
        mock_backend = get_mock_backend(storage_manager)
        assert mock_backend is not None, "Expected mock_backend to be loaded"

        # Pre-populate storage with a test memory object
        test_key = create_test_key(key_id=67890)
        test_memory_obj = create_test_memory_obj()
        mock_backend.storage[test_key.chunk_hash] = test_memory_obj

        # Clear the call history before the test
        mock_backend.call_history.clear()

        # Call get via storage_manager
        # This should iterate through backends and call get_blocking
        result = storage_manager.get(test_key)

        # Verify get_blocking was called on our dynamic backend
        assert any(
            "get_blocking:67890" in call for call in mock_backend.call_history
        ), (
            f"Expected 'get_blocking:67890' in call history, "
            f"got: {mock_backend.call_history}"
        )

        # The result should be our test memory object
        assert result is test_memory_obj, f"Expected test_memory_obj, got: {result}"

    def test_dynamic_backend_remove_called_via_storage_manager(
        self, storage_manager_with_mock_backend
    ):
        """
        Test that StorageManager can remove data through the dynamic backend.
        """
        storage_manager = storage_manager_with_mock_backend

        # Get the mock backend instance
        mock_backend = get_mock_backend(storage_manager)
        assert mock_backend is not None, "Expected mock_backend to be loaded"

        # Pre-populate storage with a test memory object
        test_key = create_test_key(key_id=11111)
        test_memory_obj = create_test_memory_obj()
        mock_backend.storage[test_key.chunk_hash] = test_memory_obj

        # Clear the call history before the test
        mock_backend.call_history.clear()

        # Call remove via storage_manager
        num_removed = storage_manager.remove(test_key)

        # Verify remove was called on our dynamic backend
        assert any("remove:11111" in call for call in mock_backend.call_history), (
            f"Expected 'remove:11111' in call history, got: {mock_backend.call_history}"
        )

        # Should have removed from at least one backend
        assert num_removed >= 1, f"Expected at least 1 removal, got: {num_removed}"

        # Verify the key is no longer in storage
        assert test_key.chunk_hash not in mock_backend.storage, (
            f"Expected key {test_key.chunk_hash} to be removed from storage"
        )

    def test_dynamic_backend_batched_contains_called_via_storage_manager(
        self, storage_manager_with_mock_backend
    ):
        """
        Test that StorageManager calls batched_contains on the dynamic backend.
        """
        storage_manager = storage_manager_with_mock_backend

        # Get the mock backend instance
        mock_backend = get_mock_backend(storage_manager)
        assert mock_backend is not None, "Expected mock_backend to be loaded"

        # Pre-populate storage with test memory objects
        test_keys = [create_test_key(key_id=i) for i in range(1000, 1003)]
        for key in test_keys:
            mock_backend.storage[key.chunk_hash] = create_test_memory_obj()

        # Clear the call history before the test
        mock_backend.call_history.clear()

        # Call batched_contains via storage_manager
        hit_chunks, block_mapping = storage_manager.batched_contains(test_keys)

        # Verify contains was called for the keys
        # (batched_contains internally calls contains for each key)
        contains_calls = [
            call for call in mock_backend.call_history if call.startswith("contains:")
        ]
        assert len(contains_calls) >= 1, (
            f"Expected at least 1 contains call, got: {mock_backend.call_history}"
        )

        # Should have found some hits in the mock backend
        assert hit_chunks >= 1, f"Expected at least 1 hit, got: {hit_chunks}"

    def test_dynamic_backend_without_storage_plugins_config(self, async_loop):
        """
        Test that no dynamic backend is loaded when storage_plugins is not configured.
        """
        # Configure extra_config but don't set storage_plugins
        config = create_test_config(
            extra_config=MOCK_BACKEND_EXTRA_CONFIG,
            storage_plugins=None,  # No storage_plugins
        )
        metadata = create_test_metadata()

        # Create storage backends
        storage_backends = CreateStorageBackends(
            config=config,
            metadata=metadata,
            loop=async_loop,
            dst_device="cpu",
        )

        # Verify no mock_backend is loaded
        assert "mock_backend" not in storage_backends, (
            f"Expected 'mock_backend' not to be in storage_backends, "
            f"got: {list(storage_backends.keys())}"
        )

        # Close all backends
        for backend in storage_backends.values():
            backend.close()

    def test_dynamic_backend_with_invalid_module_path(self, async_loop):
        """
        Test that invalid module path is handled gracefully.
        """
        # Configure with invalid module path
        extra_config = {
            "storage_plugin.invalid_backend.module_path": ("nonexistent.module.path"),
            "storage_plugin.invalid_backend.class_name": "NonexistentClass",
        }
        storage_plugins = ["invalid_backend"]

        config = create_test_config(
            extra_config=extra_config,
            storage_plugins=storage_plugins,
        )
        metadata = create_test_metadata()

        # Create storage backends - should not raise exception
        storage_backends = CreateStorageBackends(
            config=config,
            metadata=metadata,
            loop=async_loop,
            dst_device="cpu",
        )

        # Invalid backend should not be loaded
        assert "invalid_backend" not in storage_backends, (
            f"Expected 'invalid_backend' not to be in storage_backends, "
            f"got: {list(storage_backends.keys())}"
        )

        # Close all backends
        for backend in storage_backends.values():
            backend.close()
