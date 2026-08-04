# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Add import for mock
from unittest import mock
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class MockConnector(RemoteConnector):
    def __init__(self):
        self.storage = {}

    async def exists(self, key):
        return key in self.storage

    def exists_sync(self, key):
        return key in self.storage

    async def put(self, key, value):
        self.storage[key] = value

    async def get(self, key):
        return self.storage.get(key)

    async def close(self):
        pass

    async def list(self) -> List[str]:
        return []


# Mock the entire torch.cuda.Stream class
@mock.patch("torch.cuda.Stream")
def test_remote_mla_worker_id_as0(mock_stream):
    # Create configuration
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=5.0,
        local_disk=None,
        max_local_disk_size=0.0,
        remote_url="lm://localhost:65432",
        remote_serde="naive",
        use_layerwise=False,
        save_decode_cache=False,
        enable_blending=False,
        extra_config={"remote_enable_mla_worker_id_as0": True},
    )

    metadata = LMCacheMetadata(
        model_name="test-model",
        kv_dtype=torch.float16,
        kv_shape=(32, 1, 256, 64, 128),
        use_mla=True,
        world_size=4,
        local_world_size=4,
        worker_id=2,
        local_worker_id=2,
    )
    metadata0 = LMCacheMetadata(
        model_name="test-model",
        kv_dtype=torch.float16,
        kv_shape=(32, 1, 256, 64, 128),
        use_mla=True,
        world_size=4,
        local_world_size=4,
        worker_id=0,
        local_worker_id=0,
    )

    # Create memory allocator and local backend
    # First Party
    from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import (
        AdHocMemoryAllocator,
    )

    pin_allocator = AdHocMemoryAllocator()
    local_cpu_backend = LocalCPUBackend(
        config, metadata, memory_allocator=pin_allocator
    )

    loop = asyncio.new_event_loop()
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
    )
    backend.connection = MockConnector()

    # Start the event loop in a separate thread
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # Create key
    key = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=2,
        chunk_hash="test_hash",
        dtype=torch.float32,
    )

    local_cpu_backend0 = LocalCPUBackend(
        config, metadata0, memory_allocator=pin_allocator
    )
    backend0 = RemoteBackend(
        config=config,
        metadata=metadata0,
        loop=loop,
        local_cpu_backend=local_cpu_backend0,
    )
    backend0.connection = backend.connection
    # Create key
    key0 = CacheEngineKey(
        model_name="test-model",
        world_size=4,
        worker_id=0,
        chunk_hash="test_hash",
        dtype=torch.float32,
    )

    # Test not contains before adding data
    assert not backend.contains(key)
    assert not backend0.contains(key0)

    # Test submit_put_task
    memory_obj = local_cpu_backend.allocate(torch.Size([10, 10]), torch.float32)
    future = backend.submit_put_task(key, memory_obj)
    # Wait for put task to complete
    if future is not None:
        future.result()

    # Test not contains after adding data since worker_id 2 skipped put
    assert not backend.contains(key)

    future = backend0.submit_put_task(key0, memory_obj)
    # Wait for put task to complete
    if future is not None:
        future.result()

    # Test contains after adding data since worker_id 0 should put
    assert backend0.contains(key0)
    # Test contains after adding data since we use worker_id 0 instead
    assert backend.contains(key)

    # Test get_blocking
    retrieved = backend.get_blocking(key)
    assert retrieved is not None
    assert retrieved.get_shape() == torch.Size([10, 10])

    # Cleanup
    async def shutdown():
        # Get all tasks
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        # Wait for all tasks to be cancelled (or completed)
        await asyncio.gather(*tasks, return_exceptions=True)
        # Then stop the loop
        loop.stop()

    # Schedule the shutdown coroutine in the event loop thread
    future = asyncio.run_coroutine_threadsafe(shutdown(), loop)
    try:
        # Wait for the shutdown to complete, but with a timeout
        future.result(timeout=10)
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        # Wait for the loop thread to finish
        loop_thread.join(timeout=1.0)
        loop.close()
