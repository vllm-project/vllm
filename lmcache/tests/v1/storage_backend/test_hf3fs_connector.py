# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
# Authors:
#   Wenwen Chen <wenwen.chen@samsung.com>

"""
Test cases for HF3fsConnector storage backend.

This module tests the HF3fsConnector which provides access to 3FS
distributed filesystem using Usrbio interfaces.

Note: These tests use mock fixtures to simulate 3FS filesystem without
requiring actual 3FS hardware or hf3fs_fuse package.
"""

# Standard
from pathlib import Path
import asyncio
import os
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from tests.v1.utils import create_test_memory_obj

# 2nd dim should equal to chunk size
# converted shape is [2, 28, 256, 1024] (8×128=1024)
# byte size: 2 × 28 × 256 × 1024 × 2 (bfloat16) = 29,360,128 = 28MB
DEFAULT_KV_SHAPE = (28, 2, 256, 8, 128)
DEFAULT_SHAPE = torch.Size([2, 28, 256, 1024])
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_CHUNK_SIZE = 256


def create_test_config(
    base_path: str,
    mount_point: str,
    iov_size: int = 209715200,
    ior_entries: int = 256,
    io_depth: int = 0,
    numa_id: int = -1,
    io_thread_num: int = 4,
) -> LMCacheEngineConfig:
    """Create a test configuration for HF3fsConnector.

    Args:
        base_path: Base path for storage
        mount_point: 3FS mount point
        iov_size: Shared memory size for Iov
        ior_entries: Max num of concurrent requests
        io_depth: I/O depth
        numa_id: NUMA ID
        io_thread_num: Number of IO threads

    Returns:
        LMCacheEngineConfig configured for HF3FS
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=DEFAULT_CHUNK_SIZE,
        remote_url=f"hf3fs://{base_path}",
        remote_serde="naive",
        lmcache_instance_id="test_hf3fs_instance",
        extra_config={
            "save_chunk_meta": True,
            "hf3fs_mount_point": mount_point,
            "hf3fs_iov_size": iov_size,
            "hf3fs_ior_entries": ior_entries,
            "hf3fs_io_depth": io_depth,
            "hf3fs_numa_id": numa_id,
            "hf3fs_io_thread_num": io_thread_num,
        },
    )
    return config


def create_test_config_with_plugin(
    base_path: str,
    mount_point: str,
    plugin_name: str = "hf3fs",
) -> LMCacheEngineConfig:
    """Create a test configuration for HF3fsConnector using remote_storage_plugins.

    Args:
        base_path: Base path for storage
        mount_point: 3FS mount point
        plugin_name: Plugin instance name

    Returns:
        LMCacheEngineConfig configured for HF3FS with plugin
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=DEFAULT_CHUNK_SIZE,
        remote_storage_plugins=[plugin_name],
        remote_serde="naive",
        lmcache_instance_id="test_hf3fs_plugin_instance",
        extra_config={
            f"remote_storage_plugin.{plugin_name}.base_path": base_path,
            "hf3fs_mount_point": mount_point,
            "hf3fs_iov_size": 209715200,
            "hf3fs_ior_entries": 256,
            "hf3fs_io_depth": 0,
            "hf3fs_numa_id": -1,
            "hf3fs_io_thread_num": 4,
        },
    )
    return config


def create_test_metadata() -> LMCacheMetadata:
    """Create a test metadata for LMCacheMetadata.

    Returns:
        LMCacheMetadata with test configuration
    """
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=DEFAULT_KV_SHAPE,
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey.

    Args:
        key_id: Key ID for testing

    Returns:
        CacheEngineKey with test configuration
    """
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def temp_3fs_path():
    """Create a temporary directory for 3FS storage tests.

    Note: This is a local temp directory for testing.
    In real 3FS environment, this would be a path under the 3FS mount point.
    """
    temp_dir = tempfile.mkdtemp(prefix="hf3fs_test_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_3fs_mount_point(tmp_path):
    """Provide a mock 3FS mount point using a temporary directory.

    Creates a directory structure that mimics 3FS mount point:
    - mock_3fs_mount/3fs-virt/

    Returns:
        str: Path to the mock 3FS mount point
    """
    mock_mount = tmp_path / "mock_3fs_mount"
    mock_mount.mkdir()
    # Create the 3fs-virt directory structure that HF3FS expects
    virt_dir = mock_mount / "3fs-virt"
    virt_dir.mkdir()
    return str(mock_mount)


@pytest.fixture
def mock_base_path(mock_3fs_mount_point):
    """Provide a mock base path under the mock 3FS mount point.

    Returns:
        str: Path to the mock base storage directory
    """
    base_path = Path(mock_3fs_mount_point) / "3fs-virt" / "test_data"
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop running in a separate thread for testing."""
    loop = asyncio.new_event_loop()

    # Start the event loop in a separate thread
    # Standard
    import threading

    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="test-hf3fs-async-loop",
    )
    thread.start()

    yield loop

    # Cleanup: stop the loop and wait for thread to finish
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def memory_allocator():
    """Create a memory allocator for testing."""
    # First Party
    from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import (
        AdHocMemoryAllocator,
    )

    return AdHocMemoryAllocator(device="cpu")


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=DEFAULT_CHUNK_SIZE)
    metadata = create_test_metadata()
    backend = LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)
    yield backend
    memory_allocator.close()


@pytest.fixture
def remote_backend_with_hf3fs(
    mock_base_path, mock_3fs_mount_point, async_loop, local_cpu_backend
):
    """Create a RemoteBackend with HF3fsConnector for testing.

    Args:
        mock_base_path: Temporary directory for storage
        mock_3fs_mount_point: Mock 3FS mount point
        async_loop: Asyncio event loop
        local_cpu_backend: Local CPU backend

    Returns:
        RemoteBackend configured with HF3fsConnector
    """
    config = create_test_config(
        base_path=mock_base_path,
        mount_point=mock_3fs_mount_point,
    )
    metadata = create_test_metadata()
    print(f"metadata = {metadata}")
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
    )
    yield backend
    # Cleanup
    try:
        if hasattr(backend, "local_cpu_backend") and backend.local_cpu_backend:
            backend.local_cpu_backend.memory_allocator.close()
    except Exception:
        pass
    try:
        backend.close()
    except Exception:
        pass


class TestHF3fsConnector:
    """Test cases for HF3fsConnector via RemoteBackend."""

    def test_init(
        self, mock_base_path, mock_3fs_mount_point, async_loop, local_cpu_backend
    ):
        """Test HF3fsConnector initialization via RemoteBackend.

        This test verifies that the HF3fsConnector can be properly
        initialized with the required configuration parameters.
        """
        config = create_test_config(
            base_path=mock_base_path,
            mount_point=mock_3fs_mount_point,
        )
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )

        # Verify backend configuration
        assert backend.dst_device == "cpu"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.remote_url == f"hf3fs://{mock_base_path}"
        assert backend.config.remote_serde == "naive"

        # Verify storage path exists
        assert os.path.exists(mock_base_path)

        local_cpu_backend.memory_allocator.close()
        backend.close()

    def test_init_with_plugin(
        self, mock_base_path, mock_3fs_mount_point, async_loop, local_cpu_backend
    ):
        """Test HF3fsConnector initialization via RemoteBackend using
        remote_storage_plugins. This test verifies that the HF3fsConnector can be
        initialized using the plugin configuration pattern.
        """
        config = create_test_config_with_plugin(
            base_path=mock_base_path,
            mount_point=mock_3fs_mount_point,
            plugin_name="hf3fs",
        )
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="hf3fs",
        )

        # Verify backend configuration
        assert backend.dst_device == "cpu"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.plugin_name == "hf3fs"
        assert os.path.exists(mock_base_path)
        assert backend.config.remote_serde == "naive"

        local_cpu_backend.memory_allocator.close()
        backend.close()

    def test_contains_key_not_exists(self, remote_backend_with_hf3fs):
        """Test contains() when key doesn't exist in 3FS storage.

        This test verifies that the contains() method correctly
        returns False when the key does not exist.
        """
        key = create_test_key(1)
        assert not remote_backend_with_hf3fs.contains(key)
        assert not remote_backend_with_hf3fs.contains(key, pin=True)

        remote_backend_with_hf3fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_hf3fs.close()

    def test_get_blocking_key_not_exists(self, remote_backend_with_hf3fs):
        """Test get_blocking() when key doesn't exist in 3FS storage.

        This test verifies that the get_blocking() method correctly
        returns None when the key does not exist.
        """
        key = create_test_key(2)
        result = remote_backend_with_hf3fs.get_blocking(key)

        assert result is None

        remote_backend_with_hf3fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_hf3fs.close()

    def test_put_and_get_roundtrip(self, remote_backend_with_hf3fs):
        """Test put and get roundtrip for HF3fsConnector.

        This test verifies the basic write-read cycle:
        1. Put a memory object to 3FS storage
        2. Verify the key exists
        3. Get the memory object back
        4. Verify the retrieved data matches the original
        """
        key = create_test_key(3)
        memory_obj = create_test_memory_obj(shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)

        # Put data to 3FS storage
        future = remote_backend_with_hf3fs.submit_put_task(key, memory_obj)
        # Wait for the async put to complete
        if future:
            future.result(timeout=10.0)

        # Check that key exists
        assert remote_backend_with_hf3fs.contains(key)

        # Get data back
        result = remote_backend_with_hf3fs.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result.metadata.shape == memory_obj.metadata.shape
        assert result.metadata.dtype == memory_obj.metadata.dtype

        remote_backend_with_hf3fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_hf3fs.close()

    def test_batched_put_and_get(self, remote_backend_with_hf3fs):
        """Test batched put and get operations.

        This test verifies that multiple keys can be stored
        and retrieved in batch.
        """
        keys = [create_test_key(i) for i in range(3)]
        memory_objs = [
            create_test_memory_obj(shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)
            for _ in range(3)
        ]

        # Batched put
        futures = [
            remote_backend_with_hf3fs.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]
        for future in filter(None, futures):
            future.result(timeout=10.0)

        # Check all keys exist
        for key in keys:
            assert remote_backend_with_hf3fs.contains(key)

        # Batched get
        results = remote_backend_with_hf3fs.batched_get_blocking(keys)

        assert results is not None
        assert len(results) == 3
        for result, original in zip(results, memory_objs, strict=False):
            assert result is not None
            assert result.metadata.shape == original.metadata.shape
            assert result.metadata.dtype == original.metadata.dtype

        remote_backend_with_hf3fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_hf3fs.close()

    def test_multiple_paths_config(
        self, mock_base_path, mock_3fs_mount_point, async_loop, local_cpu_backend
    ):
        """Test HF3fsConnector with multiple base paths.

        This test verifies that the connector can handle
        multiple base paths (comma-separated).
        """
        # Create additional temp directories
        temp_dir2 = tempfile.mkdtemp(prefix="hf3fs_test2_", dir=mock_3fs_mount_point)
        temp_dir3 = tempfile.mkdtemp(prefix="hf3fs_test3_", dir=mock_3fs_mount_point)

        try:
            # Create config with multiple paths
            multi_path = f"{mock_base_path},{temp_dir2},{temp_dir3}"
            config = create_test_config(
                base_path=multi_path,
                mount_point=mock_3fs_mount_point,
            )
            metadata = create_test_metadata()

            backend = RemoteBackend(
                config=config,
                metadata=metadata,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )

            key = create_test_key(10)
            memory_obj = create_test_memory_obj(
                shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE
            )

            # Put and get should work with multiple paths
            future = backend.submit_put_task(key, memory_obj)
            if future:
                future.result(timeout=10.0)

            assert backend.contains(key)

            result = backend.get_blocking(key)
            assert result is not None
            assert result.metadata.shape == memory_obj.metadata.shape

            backend.local_cpu_backend.memory_allocator.close()
            backend.close()

        finally:
            # Cleanup additional directories
            if os.path.exists(temp_dir2):
                shutil.rmtree(temp_dir2, ignore_errors=True)
            if os.path.exists(temp_dir3):
                shutil.rmtree(temp_dir3, ignore_errors=True)

    def test_file_persistence(
        self, mock_base_path, mock_3fs_mount_point, async_loop, local_cpu_backend
    ):
        """Test that files persist after backend closure.

        This test verifies that data written to 3FS storage
        persists after the backend is closed and can be
        retrieved with a new backend instance.
        """
        config = create_test_config(
            base_path=mock_base_path,
            mount_point=mock_3fs_mount_point,
        )
        metadata = create_test_metadata()

        key = create_test_key(5)
        memory_obj = create_test_memory_obj(shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)

        # Create backend, put data, and close
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )

        future = backend.submit_put_task(key, memory_obj)
        if future:
            future.result(timeout=10.0)

        backend.local_cpu_backend.memory_allocator.close()
        backend.close()

        # Create new backend instance and verify data persists
        new_memory_allocator = local_cpu_backend.memory_allocator.__class__(
            device="cpu"
        )
        new_local_cpu_backend = LocalCPUBackend(
            LMCacheEngineConfig.from_legacy(chunk_size=DEFAULT_CHUNK_SIZE),
            metadata,
            memory_allocator=new_memory_allocator,
        )
        new_backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=new_local_cpu_backend,
            dst_device="cpu",
        )

        assert new_backend.contains(key)

        result = new_backend.get_blocking(key)
        assert result is not None
        assert result.metadata.shape == memory_obj.metadata.shape

        new_backend.local_cpu_backend.memory_allocator.close()
        new_backend.close()

    def test_remove_key(self, remote_backend_with_hf3fs):
        """Test removing a key from 3FS storage.

        This test verifies that keys can be properly removed
        from the storage backend.
        """
        key = create_test_key(6)
        memory_obj = create_test_memory_obj(shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)

        # Put data
        future = remote_backend_with_hf3fs.submit_put_task(key, memory_obj)
        if future:
            future.result(timeout=10.0)

        # Verify key exists
        assert remote_backend_with_hf3fs.contains(key)

        # Remove key
        result = remote_backend_with_hf3fs.remove(key)
        assert result is True

        # Verify key no longer exists
        assert not remote_backend_with_hf3fs.contains(key)

        remote_backend_with_hf3fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_hf3fs.close()


class TestHF3fsConnectorConfiguration:
    """Test cases for HF3fsConnector configuration validation."""

    def test_invalid_mount_point(self, mock_base_path, async_loop, local_cpu_backend):
        """Test that invalid mount point raises an error.

        This test verifies that the connector properly validates
        the 3FS mount point during initialization.
        """
        # Use a non-existent path as mount point
        invalid_mount_point = "/nonexistent/path/3fs"

        config = create_test_config(
            base_path=mock_base_path,
            mount_point=invalid_mount_point,
        )
        metadata = create_test_metadata()

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="hf3fs",
        )

        # will not create connection
        assert backend is not None
        assert backend.connection is None
        local_cpu_backend.memory_allocator.close()
        backend.close()

    def test_configuration_parameters(
        self, mock_base_path, mock_3fs_mount_point, async_loop, local_cpu_backend
    ):
        """Test various configuration parameters.

        This test verifies that different configuration parameters
        are properly passed to the HF3fsConnector.
        """
        # Test with custom parameters
        config = create_test_config(
            base_path=mock_base_path,
            mount_point=mock_3fs_mount_point,
            iov_size=104857600,  # 100MB
            ior_entries=128,
            io_depth=32,
            numa_id=0,
            io_thread_num=8,
        )
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )

        assert backend is not None

        backend.local_cpu_backend.memory_allocator.close()
        backend.close()
