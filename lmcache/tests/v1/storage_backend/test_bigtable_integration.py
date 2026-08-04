# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved

# Standard
import asyncio
import threading

# Third Party
from google.cloud.bigtable import Client
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from tests.v1.utils import create_test_memory_obj


# Simple test helpers
def create_test_metadata(kv_shape=(2, 2, 256, 8, 128)):
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        use_mla=False,
        role="worker",
    )


def create_test_config(extra_overrides=None):
    extras = {
        "bigtable_project_id": "test-project",
        "bigtable_instance_id": "test-instance",
        "bigtable_table_name": "test-table",
        "bigtable_family_name": "cf",
        "bigtable_column_name": "data",
        "bigtable_max_chunk_size_mb": 5.0,  # 5MB threshold for testing skip logic
        "bigtable_max_retries": 2,
    }
    if extra_overrides:
        extras.update(extra_overrides)

    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_storage_plugins=["bigtable"],
        remote_serde="naive",
        lmcache_instance_id="test_instance",
        extra_config=extras,
    )


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
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
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


@pytest.mark.integration
class TestBigtableEmulatorIntegration:
    @pytest.fixture(autouse=True)
    def setup_emulator_table(self, bigtable_emulator):
        """Prepare instance, table, and column family in the emulator."""
        project_id = "test-project"
        instance_id = "test-instance"
        table_name = "test-table"
        family_name = "cf"

        # Initialize sync admin client using the emulator host
        client = Client(project=project_id, admin=True)
        instance = client.instance(instance_id)

        table = instance.table(table_name)
        try:
            if table.exists():
                table.delete()
        except Exception:
            pass

        table.create()
        cf = table.column_family(family_name)
        cf.create()

        yield

        # Clean up table after each test
        try:
            if table.exists():
                table.delete()
        except Exception:
            pass

    @pytest.fixture
    def memory_allocator(self):
        alloc = MixedMemoryAllocator(100 * 1024 * 1024)  # 100MB
        yield alloc
        alloc.close()

    @pytest.fixture
    def local_cpu_backend(self, memory_allocator):
        config = LMCacheEngineConfig.from_defaults(chunk_size=256)
        metadata = create_test_metadata()
        backend = LocalCPUBackend(config, metadata, memory_allocator=memory_allocator)
        yield backend
        backend.close()

    def test_integration_put_and_get(self, async_loop, local_cpu_backend):
        """Verify standard put and get of chunk bytes with emulator."""
        config = create_test_config()
        metadata = create_test_metadata()

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = CacheEngineKey("test_model", 0, 0, 256, torch.bfloat16)

        # Use test helper to create concrete MemoryObj and fill it with values
        memory_obj = create_test_memory_obj(
            shape=torch.Size([2, 2, 256, 8, 128]), dtype=torch.bfloat16
        )
        memory_obj.tensor.fill_(3.14)

        # Write asynchronously and wait for future
        fut = backend.submit_put_task(key, memory_obj)
        fut.result(timeout=5.0)

        # Query contains
        assert backend.contains(key)

        # Read back using get_blocking (allocates automatically)
        retrieved_obj = backend.get_blocking(key)
        assert retrieved_obj is not None
        assert torch.all(retrieved_obj.tensor == 3.14)

        backend.close()

    def test_integration_batched_put_and_get(self, async_loop, local_cpu_backend):
        """Verify dynamic batching and batched operations with emulator."""
        config = create_test_config()
        metadata = create_test_metadata()

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        keys = [
            CacheEngineKey("test_model", 0, i, 256, torch.bfloat16) for i in range(5)
        ]

        memory_objs = []
        for i in range(5):
            memory_obj = create_test_memory_obj(
                shape=torch.Size([2, 2, 256, 8, 128]), dtype=torch.bfloat16
            )
            memory_obj.tensor.fill_(float(i))
            memory_objs.append(memory_obj)

        # Use threading events to wait for batched write completion
        done_events = [threading.Event() for _ in range(5)]

        def on_complete(key):
            idx = keys.index(key)
            done_events[idx].set()

        # Batched Put
        backend.batched_submit_put_task(
            keys, memory_objs, on_complete_callback=on_complete
        )

        # Wait for all writes to finish
        for ev in done_events:
            assert ev.wait(timeout=10.0)

        # Assert all keys are in backend
        for key in keys:
            assert backend.contains(key)

        # Read all back using batched_get_blocking
        retrieved_objs = backend.batched_get_blocking(keys)
        assert len(retrieved_objs) == 5

        for i in range(5):
            assert retrieved_objs[i] is not None
            assert torch.all(retrieved_objs[i].tensor == float(i))

        backend.close()

    def test_integration_remove(self, async_loop, local_cpu_backend):
        """Verify deleting chunks from emulator works."""
        config = create_test_config()
        metadata = create_test_metadata()

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = CacheEngineKey("test_model", 0, 10, 256, torch.bfloat16)
        memory_obj = create_test_memory_obj(
            shape=torch.Size([2, 2, 256, 8, 128]), dtype=torch.bfloat16
        )
        memory_obj.tensor.fill_(1.0)

        fut = backend.submit_put_task(key, memory_obj)
        fut.result(timeout=5.0)
        assert backend.contains(key)

        # Remove
        assert backend.remove(key)
        assert not backend.contains(key)

        backend.close()

    def test_integration_skips_large_writes(self, async_loop, local_cpu_backend):
        """Verify writing a chunk larger than max_chunk_size_mb is skipped
        without failure.
        """
        # Max size is 5.0 MB, sharding disabled
        config = create_test_config(extra_overrides={"bigtable_layer_group_size": 0})
        metadata = create_test_metadata()

        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            plugin_name="bigtable",
        )

        key = CacheEngineKey("test_model", 0, 99, 256, torch.bfloat16)

        # Create a payload of ~8.38 MB (exceeds the 5.0 MB threshold)
        # 8 * 2 * 256 * 8 * 128 * 2 bytes = 8,388,608 bytes
        memory_obj = create_test_memory_obj(
            shape=torch.Size([8, 2, 256, 8, 128]), dtype=torch.bfloat16
        )

        # Try to put and wait for it
        fut = backend.submit_put_task(key, memory_obj)
        fut.result(timeout=5.0)

        # Verify it was NOT written (skips write)
        assert not backend.contains(key)

        backend.close()
