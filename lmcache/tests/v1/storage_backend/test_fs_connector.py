# SPDX-License-Identifier: Apache-2.0
# Standard
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


def create_test_config(fs_path: str):
    """Create a test configuration for FSConnector."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_url=f"fs://host:0/{fs_path}",
        remote_serde="naive",
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_config_with_plugin(fs_path: str):
    """Create a test configuration for FSConnector using remote_storage_plugins."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_storage_plugins=["fs"],
        remote_serde="naive",
        lmcache_instance_id="test_instance",
        extra_config={
            "remote_storage_plugin.fs.base_path": fs_path,
        },
    )
    return config


def create_test_config_with_dual_plugins(fs_path1: str, fs_path2: str):
    """Create config with two fs_connector instances."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_storage_plugins=["fs.primary", "fs.backup"],
        remote_serde="naive",
        lmcache_instance_id="test_instance",
        extra_config={
            "remote_storage_plugin.fs.primary.base_path": fs_path1,
            "remote_storage_plugin.fs.backup.base_path": fs_path2,
        },
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def temp_fs_path():
    """Create a temporary directory for filesystem storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


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
        name="test-async-loop",
    )
    thread.start()

    yield loop

    # Cleanup: stop the loop and wait for thread to finish
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    metadata = create_test_metadata()
    return LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)


@pytest.fixture
def remote_backend_with_fs(temp_fs_path, async_loop, local_cpu_backend):
    """Create a RemoteBackend with FSConnector for testing."""
    config = create_test_config(temp_fs_path)
    metadata = create_test_metadata()
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
    )
    yield backend
    backend.local_cpu_backend.memory_allocator.close()
    backend.close()


class TestFSConnector:
    """Test cases for FSConnector via RemoteBackend."""

    def test_init(self, temp_fs_path, async_loop, local_cpu_backend):
        """Test FSConnector initialization via RemoteBackend."""
        config = create_test_config(temp_fs_path)
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )

        assert backend.dst_device == "cpu"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.remote_url == f"fs://host:0/{temp_fs_path}"
        assert os.path.exists(temp_fs_path)
        assert backend.config.remote_serde == "naive"

        local_cpu_backend.memory_allocator.close()
        backend.close()

    def test_init_with_plugin(self, temp_fs_path, async_loop, local_cpu_backend):
        """Test FSConnector init via RemoteBackend
        using remote_storage_plugins."""
        config = create_test_config_with_plugin(temp_fs_path)
        metadata = create_test_metadata()
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="fs",
        )

        assert backend.dst_device == "cpu"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.plugin_name == "fs"
        assert os.path.exists(temp_fs_path)
        assert backend.config.remote_serde == "naive"

        local_cpu_backend.memory_allocator.close()
        backend.close()

    def test_dual_fs_instances(self, async_loop, local_cpu_backend):
        """Test two fs_connector instances with different paths."""
        dir1 = tempfile.mkdtemp()
        dir2 = tempfile.mkdtemp()
        try:
            config = create_test_config_with_dual_plugins(dir1, dir2)
            metadata = create_test_metadata()

            backend1 = RemoteBackend(
                config=config,
                metadata=metadata,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
                plugin_name="fs.primary",
            )
            backend2 = RemoteBackend(
                config=config,
                metadata=metadata,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
                plugin_name="fs.backup",
            )

            key = create_test_key(99)
            memory_obj = create_test_memory_obj()

            # Put to backend1 only
            future = backend1.submit_put_task(key, memory_obj)
            if future:
                future.result(timeout=5.0)

            # backend1 has the key, backend2 does not
            assert backend1.contains(key)
            assert not backend2.contains(key)

            # Put to backend2 as well
            future2 = backend2.submit_put_task(key, memory_obj)
            if future2:
                future2.result(timeout=5.0)

            assert backend2.contains(key)

            backend1.close()
            backend2.close()
            local_cpu_backend.memory_allocator.close()
        finally:
            if os.path.exists(dir1):
                shutil.rmtree(dir1)
            if os.path.exists(dir2):
                shutil.rmtree(dir2)

    def test_contains_key_not_exists(self, remote_backend_with_fs):
        """Test contains() when key doesn't exist in filesystem."""
        key = create_test_key(1)
        assert not remote_backend_with_fs.contains(key)
        assert not remote_backend_with_fs.contains(key, pin=True)

        remote_backend_with_fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_fs.close()

    def test_get_blocking_key_not_exists(self, remote_backend_with_fs):
        """Test get_blocking() when key doesn't exist in filesystem."""
        key = create_test_key(2)
        result = remote_backend_with_fs.get_blocking(key)

        assert result is None

        remote_backend_with_fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_fs.close()

    def test_put_and_get_roundtrip(self, remote_backend_with_fs):
        """Test put and get roundtrip for FSConnector."""
        key = create_test_key(3)
        memory_obj = create_test_memory_obj()

        # Put data to filesystem
        future = remote_backend_with_fs.submit_put_task(key, memory_obj)
        # Wait for the async put to complete
        if future:
            future.result(timeout=5.0)

        # Check that key exists
        assert remote_backend_with_fs.contains(key)

        # Get data back
        result = remote_backend_with_fs.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result.metadata.shape == memory_obj.metadata.shape
        assert result.metadata.dtype == memory_obj.metadata.dtype

        remote_backend_with_fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_fs.close()

    def test_batched_put_and_get(self, remote_backend_with_fs):
        """Test batched put and get operations."""
        keys = [create_test_key(i) for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        # Batched put
        futures = [
            remote_backend_with_fs.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]
        for future in filter(None, futures):
            future.result(timeout=5.0)

        # Check all keys exist
        for key in keys:
            assert remote_backend_with_fs.contains(key)

        # Batched get
        results = remote_backend_with_fs.batched_get_blocking(keys)

        assert results is not None
        assert len(results) == 3
        for result, original in zip(results, memory_objs, strict=False):
            assert result is not None
            assert result.metadata.shape == original.metadata.shape
            assert result.metadata.dtype == original.metadata.dtype

        remote_backend_with_fs.local_cpu_backend.memory_allocator.close()
        remote_backend_with_fs.close()

    def test_multiple_paths_config(self, temp_fs_path, async_loop, local_cpu_backend):
        """Test FSConnector with multiple paths."""
        # Create additional temp directories
        temp_dir2 = tempfile.mkdtemp()
        temp_dir3 = tempfile.mkdtemp()

        try:
            # Create config with multiple paths
            multi_path = f"{temp_fs_path},{temp_dir2},{temp_dir3}"
            config = create_test_config(multi_path)
            metadata = create_test_metadata()

            backend = RemoteBackend(
                config=config,
                metadata=metadata,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )

            key = create_test_key(10)
            memory_obj = create_test_memory_obj()

            # Put and get should work with multiple paths
            future = backend.submit_put_task(key, memory_obj)
            if future:
                future.result(timeout=5.0)

            assert backend.contains(key)

            result = backend.get_blocking(key)
            assert result is not None
            assert result.metadata.shape == memory_obj.metadata.shape

            backend.local_cpu_backend.memory_allocator.close()
            backend.close()

        finally:
            # Cleanup additional directories
            if os.path.exists(temp_dir2):
                shutil.rmtree(temp_dir2)
            if os.path.exists(temp_dir3):
                shutil.rmtree(temp_dir3)

    def test_file_persistence(self, temp_fs_path, async_loop, local_cpu_backend):
        """Test that files persist after backend closure."""
        config = create_test_config(temp_fs_path)
        metadata = create_test_metadata()

        key = create_test_key(5)
        memory_obj = create_test_memory_obj()

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
            future.result(timeout=5.0)

        backend.local_cpu_backend.memory_allocator.close()
        backend.close()

        # Create new backend instance and verify data persists
        new_local_cpu_backend = LocalCPUBackend(
            LMCacheEngineConfig.from_legacy(chunk_size=256),
            local_cpu_backend.metadata,
            memory_allocator=local_cpu_backend.memory_allocator,
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
