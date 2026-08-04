# SPDX-License-Identifier: Apache-2.0
# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import BatchedKVOperationMsg, OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from tests.v1.utils import create_test_memory_obj
import lmcache.v1.storage_backend.local_cpu_backend as local_cpu_backend_module


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []
        self._lock = threading.Lock()

    def put_msg(self, msg):
        with self._lock:
            self.messages.append(msg)


def create_test_config(
    local_cpu: bool = True, use_layerwise: bool = False, enable_blending: bool = False
):
    """Create a test configuration for LocalCPUBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=local_cpu,
        use_layerwise=use_layerwise,
        enable_blending=enable_blending,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_key(key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def create_test_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = create_test_config()

    # Initialize PinMonitor before creating backend
    PinMonitor.GetOrCreate(config)

    backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

    yield backend

    # Cleanup: destroy PinMonitor after test
    PinMonitor.DestroyInstance()


@pytest.fixture
def local_cpu_backend_disabled(memory_allocator):
    """Create a LocalCPUBackend with local_cpu disabled."""
    config = create_test_config(local_cpu=False)

    # Initialize PinMonitor before creating backend
    PinMonitor.GetOrCreate(config)

    backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

    yield backend

    # Cleanup: destroy PinMonitor after test
    PinMonitor.DestroyInstance()


class TestLocalCPUBackend:
    """Test cases for LocalCPUBackend."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_init(self, memory_allocator):
        """Test LocalCPUBackend initialization."""
        config = create_test_config()
        backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

        assert backend.use_hot is True
        assert backend.memory_allocator == memory_allocator
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert len(backend.hot_cache) == 0
        assert backend.layerwise is False
        assert backend.enable_blending is False

        memory_allocator.close()

    def test_init_with_lookup_server_and_worker(self, memory_allocator):
        """Test LocalCPUBackend initialization with lookup server and worker."""
        config = create_test_config()
        lmcache_worker = MockLMCacheWorker()

        backend = LocalCPUBackend(
            config=config,
            memory_allocator=memory_allocator,
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        memory_allocator.close()

    def test_init_with_layerwise_config(self, memory_allocator):
        """Test LocalCPUBackend initialization with layerwise configuration."""
        config = create_test_config(use_layerwise=True, enable_blending=True)
        backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

        assert backend.layerwise is True
        assert backend.enable_blending is True

        memory_allocator.close()

    def test_str(self, local_cpu_backend):
        """Test string representation."""
        assert str(local_cpu_backend) == "LocalCPUBackend"

        local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_cpu_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key("nonexistent")
        assert not local_cpu_backend.contains(key)
        assert not local_cpu_backend.contains(key, pin=True)

        local_cpu_backend.memory_allocator.close()

    def test_contains_key_exists(self, local_cpu_backend):
        """Test contains() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        assert local_cpu_backend.contains(key)
        assert local_cpu_backend.contains(key, pin=True)

        local_cpu_backend.memory_allocator.close()

    def test_exists_in_put_tasks(self, local_cpu_backend):
        """Test exists_in_put_tasks()."""
        key = create_test_key("test_key")
        # LocalCPUBackend always returns False for exists_in_put_tasks
        assert not local_cpu_backend.exists_in_put_tasks(key)
        local_cpu_backend.memory_allocator.close()

    def test_submit_put_task(self, local_cpu_backend):
        """Test submit_put_task()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        future = local_cpu_backend.submit_put_task(key, memory_obj)

        # LocalCPUBackend returns None for submit_put_task
        assert future is None
        assert key in local_cpu_backend.hot_cache
        assert local_cpu_backend.hot_cache[key] == memory_obj
        assert (
            memory_obj.get_ref_count() == 2
        )  # 1 from creation + 1 from submit_put_task
        local_cpu_backend.memory_allocator.close()

    def test_submit_put_task_reinsert(self, local_cpu_backend):
        """Test submit_put_task() with reinsertion."""
        key = create_test_key("test_key")
        memory_obj1 = create_test_memory_obj(shape=torch.Size([2, 16, 8, 128]))
        memory_obj2 = create_test_memory_obj(shape=torch.Size([2, 32, 8, 128]))

        # First insertion
        local_cpu_backend.submit_put_task(key, memory_obj1)
        assert local_cpu_backend.hot_cache[key] == memory_obj1

        # Reinsertion
        local_cpu_backend.submit_put_task(key, memory_obj2)
        assert local_cpu_backend.hot_cache[key] != memory_obj2
        assert memory_obj1.get_ref_count() == 2
        assert memory_obj2.get_ref_count() == 1

        local_cpu_backend.memory_allocator.close()

    def test_batched_submit_put_task(self, local_cpu_backend):
        """Test batched_submit_put_task()."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        futures = local_cpu_backend.batched_submit_put_task(keys, memory_objs)

        # LocalCPUBackend returns None for batched_submit_put_task
        assert futures is None

        # Check that all keys were inserted
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            assert key in local_cpu_backend.hot_cache
            assert local_cpu_backend.hot_cache[key] == memory_obj

        local_cpu_backend.memory_allocator.close()

    def test_batched_submit_put_task_disabled(self, local_cpu_backend_disabled):
        """Test batched_submit_put_task() when local_cpu is disabled."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        futures = local_cpu_backend_disabled.batched_submit_put_task(keys, memory_objs)

        # Should return None when local_cpu is disabled
        assert futures is None

        local_cpu_backend_disabled.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_cpu_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key("nonexistent")
        result = local_cpu_backend.get_blocking(key)

        assert result is None

        local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_exists(self, local_cpu_backend):
        """Test get_blocking() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        result = local_cpu_backend.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result == memory_obj
        assert (
            result.get_ref_count() == 3
        )  # 1 from creation + 1 from submit_put_task + 1 from get_blocking

        local_cpu_backend.memory_allocator.close()

    def test_pin_unpin(self, local_cpu_backend):
        """Test pin() and unpin() operations."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        # Test pin
        assert local_cpu_backend.pin(key)
        assert memory_obj.is_pinned

        # Test unpin
        assert local_cpu_backend.unpin(key)
        assert not memory_obj.is_pinned

        # Test pin/unpin non-existent key
        non_existent_key = create_test_key("non_existent")
        assert not local_cpu_backend.pin(non_existent_key)
        assert not local_cpu_backend.unpin(non_existent_key)

        local_cpu_backend.memory_allocator.close()

    def test_remove(self, local_cpu_backend):
        """Test remove()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)
        assert key in local_cpu_backend.hot_cache

        # Remove the key
        result = local_cpu_backend.remove(key)

        assert result is True
        assert key not in local_cpu_backend.hot_cache
        assert memory_obj.get_ref_count() == 1  # Should be decremented

        local_cpu_backend.memory_allocator.close()

    def test_remove_non_existent(self, local_cpu_backend):
        """Test remove() with non-existent key."""
        key = create_test_key("nonexistent")
        result = local_cpu_backend.remove(key)

        assert result is False

        local_cpu_backend.memory_allocator.close()

    def test_remove_with_worker(self, memory_allocator, lmcache_engine_metadata):
        """Test remove() with LMCacheWorker."""
        config = create_test_config()
        lmcache_worker = MockLMCacheWorker()

        backend = LocalCPUBackend(
            config=config,
            metadata=lmcache_engine_metadata,
            memory_allocator=memory_allocator,
            lmcache_worker=lmcache_worker,
        )

        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        backend.submit_put_task(key, memory_obj)

        # Remove the key
        backend.remove(key)

        # Manually flush to ensure messages are sent for testing
        if backend.batched_msg_sender is not None:
            backend.batched_msg_sender.flush()

        # Check that we have batched messages
        batched_msgs = [
            msg
            for msg in lmcache_worker.messages
            if isinstance(msg, BatchedKVOperationMsg)
        ]
        assert len(batched_msgs) >= 1, "Should have at least one batched message"

        # Collect all operations from all batches
        all_admit_ops = []
        all_evict_ops = []
        for msg in batched_msgs:
            for op in msg.operations:
                if op.op_type == OpType.ADMIT:
                    all_admit_ops.append(op)
                elif op.op_type == OpType.EVICT:
                    all_evict_ops.append(op)

        # Verify we have exactly one ADMIT and one EVICT operation
        assert len(all_admit_ops) == 1, "Should have exactly one ADMIT operation"
        assert len(all_evict_ops) == 1, "Should have exactly one EVICT operation"

        # Verify the operations are for the correct key
        assert all_admit_ops[0].key == key.chunk_hash
        assert all_evict_ops[0].key == key.chunk_hash

        memory_allocator.close()

    def test_allocate(self, local_cpu_backend):
        """Test allocate()."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16

        memory_obj = local_cpu_backend.allocate(shape, dtype)

        assert memory_obj is not None
        assert isinstance(memory_obj, MemoryObj)
        assert memory_obj.metadata.shape == shape
        assert memory_obj.metadata.dtype == dtype

        local_cpu_backend.memory_allocator.close()

    def test_allocate_with_format(self, local_cpu_backend):
        """Test allocate() with specific format."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD

        memory_obj = local_cpu_backend.allocate(shape, dtype, fmt)

        assert memory_obj is not None
        assert memory_obj.metadata.fmt == fmt

        local_cpu_backend.memory_allocator.close()

    def test_allocate_with_layerwise_config(self, memory_allocator):
        """Test allocate() with layerwise configuration."""
        config = create_test_config(use_layerwise=True, enable_blending=True)
        backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16

        memory_obj = backend.allocate(shape, dtype)

        assert memory_obj is not None
        # Should use KV_2TD format when layerwise=True and enable_blending=True
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD

        memory_allocator.close()

    def test_batched_allocate(self, local_cpu_backend):
        """Test batched_allocate()."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        batch_size = 3

        memory_objs = local_cpu_backend.batched_allocate(shape, dtype, batch_size)

        assert memory_objs is not None
        assert len(memory_objs) == batch_size
        for memory_obj in memory_objs:
            assert isinstance(memory_obj, MemoryObj)
            assert memory_obj.metadata.shape == shape
            assert memory_obj.metadata.dtype == dtype

        local_cpu_backend.memory_allocator.close()

    def test_get_keys(self, local_cpu_backend):
        """Test get_keys()."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        # Insert keys
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            local_cpu_backend.submit_put_task(key, memory_obj)

        # Get keys
        retrieved_keys = local_cpu_backend.get_keys()

        assert len(retrieved_keys) == 3
        assert all(key in retrieved_keys for key in keys)

        local_cpu_backend.memory_allocator.close()

    def test_get_keys_empty(self, local_cpu_backend):
        """Test get_keys() when cache is empty."""
        keys = local_cpu_backend.get_keys()

        assert len(keys) == 0

        local_cpu_backend.memory_allocator.close()

    def test_concurrent_access(self, local_cpu_backend):
        """Test concurrent access to the backend."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_cpu_backend.submit_put_task(key, memory_obj)

        # Test concurrent contains() calls
        def check_contains():
            for _ in range(20):
                assert local_cpu_backend.contains(key)

        threads = [threading.Thread(target=check_contains) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        local_cpu_backend.memory_allocator.close()

    def test_thread_safety(self, local_cpu_backend):
        """Test thread safety of the backend."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_cpu_backend.submit_put_task(key, memory_obj)

        # Test concurrent operations
        def concurrent_operations():
            for _ in range(10):
                # Test contains
                local_cpu_backend.contains(key)
                # Test pin/unpin
                local_cpu_backend.pin(key)
                local_cpu_backend.unpin(key)
                # Test get_blocking
                result = local_cpu_backend.get_blocking(key)
                assert result is not None

        threads = [threading.Thread(target=concurrent_operations) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # The backend should still be in a consistent state
        assert local_cpu_backend.contains(key)

        local_cpu_backend.memory_allocator.close()

    def test_ref_count_management(self, local_cpu_backend):
        """Test reference count management."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        initial_ref_count = memory_obj.get_ref_count()

        # Insert key
        local_cpu_backend.submit_put_task(key, memory_obj)
        assert memory_obj.get_ref_count() == initial_ref_count + 1

        # Get blocking
        local_cpu_backend.get_blocking(key)
        assert memory_obj.get_ref_count() == initial_ref_count + 2

        # Remove key
        local_cpu_backend.remove(key)
        assert memory_obj.get_ref_count() == initial_ref_count + 1
        local_cpu_backend.memory_allocator.close()


@pytest.mark.no_shared_allocator
class TestLocalCPUBackendAllocatorRecovery:
    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()
        PinMonitor.DestroyInstance()

    def test_batched_allocate_fails_while_group_pinned_then_recovers(self):
        chunk_bytes = 1024 * 1024
        batch_size = 2
        shape = torch.Size([1, chunk_bytes])
        config = create_test_config()
        PinMonitor.GetOrCreate(config)
        allocator = MixedMemoryAllocator(chunk_bytes * batch_size)
        backend = LocalCPUBackend(config=config, memory_allocator=allocator)
        layer_keys = create_test_key("batched_pinned").split_layers(batch_size)

        memory_objs = backend.batched_allocate(
            shape,
            torch.uint8,
            batch_size=batch_size,
            fmt=MemoryFormat.KV_T2D,
            busy_loop=False,
        )
        assert memory_objs is not None
        backend.batched_submit_put_task(layer_keys, memory_objs)
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

        assert backend.pin(layer_keys[0])
        assert (
            backend.batched_allocate(
                shape,
                torch.uint8,
                batch_size=batch_size,
                fmt=MemoryFormat.KV_T2D,
                busy_loop=False,
            )
            is None
        )
        assert all(key in backend.hot_cache for key in layer_keys)

        assert backend.unpin(layer_keys[0])
        recovered = backend.batched_allocate(
            shape,
            torch.uint8,
            batch_size=batch_size,
            fmt=MemoryFormat.KV_T2D,
            busy_loop=False,
        )
        assert recovered is not None
        assert all(key not in backend.hot_cache for key in layer_keys)
        assert allocator.memcheck()

        for memory_obj in recovered:
            memory_obj.ref_count_down()
        allocator.close()

    def test_batched_allocate_recovers_with_fully_evictable_group(self):
        chunk_bytes = 4096
        batch_size = 2
        shape = torch.Size([1, chunk_bytes])
        config = create_test_config()
        PinMonitor.GetOrCreate(config)
        allocator = MixedMemoryAllocator(chunk_bytes * batch_size * 2)
        backend = LocalCPUBackend(config=config, memory_allocator=allocator)
        pinned_group_keys = create_test_key("batched_pinned_group").split_layers(
            batch_size
        )
        safe_group_keys = create_test_key("batched_safe_group").split_layers(batch_size)

        pinned_group_objs = backend.batched_allocate(
            shape,
            torch.uint8,
            batch_size=batch_size,
            fmt=MemoryFormat.KV_T2D,
            busy_loop=False,
        )
        safe_group_objs = backend.batched_allocate(
            shape,
            torch.uint8,
            batch_size=batch_size,
            fmt=MemoryFormat.KV_T2D,
            busy_loop=False,
        )
        assert pinned_group_objs is not None
        assert safe_group_objs is not None
        backend.batched_submit_put_task(pinned_group_keys, pinned_group_objs)
        backend.batched_submit_put_task(safe_group_keys, safe_group_objs)
        for memory_obj in pinned_group_objs + safe_group_objs:
            memory_obj.ref_count_down()

        assert backend.pin(pinned_group_keys[0])
        recovered = backend.batched_allocate(
            shape,
            torch.uint8,
            batch_size=batch_size,
            fmt=MemoryFormat.KV_T2D,
            busy_loop=False,
        )

        assert recovered is not None
        assert all(key in backend.hot_cache for key in pinned_group_keys)
        assert all(key not in backend.hot_cache for key in safe_group_keys)

        for memory_obj in recovered:
            memory_obj.ref_count_down()
        assert backend.unpin(pinned_group_keys[0])
        backend.clear()
        assert allocator.memcheck()
        allocator.close()


class TestLocalCPUBackendAllocatorAlignment:
    def test_rust_odirect_auto_alignment_for_mixed_allocator(self, monkeypatch):
        config = create_test_config(local_cpu=True)
        config.max_local_cpu_size = 0.01
        config.extra_config = {
            "rust_raw_block.device_path": "/tmp/dev.bin",
            "rust_raw_block.use_odirect": True,
            "rust_raw_block.block_align": 4096,
        }
        metadata = create_test_metadata()

        captured: dict[str, object] = {}

        class DummyMixedMemoryAllocator:
            def __init__(self, size, **kwargs):
                captured["size"] = size
                captured["kwargs"] = kwargs
                self.align_bytes = kwargs.get("align_bytes", 4096)

            def close(self):
                return None

        monkeypatch.setattr(
            local_cpu_backend_module,
            "MixedMemoryAllocator",
            DummyMixedMemoryAllocator,
        )

        backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
        try:
            kwargs = captured["kwargs"]
            assert isinstance(kwargs, dict)
            assert kwargs.get("align_bytes") == 4096
        finally:
            backend.memory_allocator.close()

    def test_explicit_alignment_override_for_mixed_allocator(self, monkeypatch):
        config = create_test_config(local_cpu=True)
        config.max_local_cpu_size = 0.01
        config.extra_config = {
            "local_cpu.pinned_align_bytes": 4096,
            "rust_raw_block.device_path": "/tmp/dev.bin",
            "rust_raw_block.use_odirect": False,
        }
        metadata = create_test_metadata()

        captured: dict[str, object] = {}

        class DummyMixedMemoryAllocator:
            def __init__(self, size, **kwargs):
                captured["size"] = size
                captured["kwargs"] = kwargs
                self.align_bytes = kwargs.get("align_bytes", 4096)

            def close(self):
                return None

        monkeypatch.setattr(
            local_cpu_backend_module,
            "MixedMemoryAllocator",
            DummyMixedMemoryAllocator,
        )

        backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
        try:
            kwargs = captured["kwargs"]
            assert isinstance(kwargs, dict)
            assert kwargs.get("align_bytes") == 4096
        finally:
            backend.memory_allocator.close()
