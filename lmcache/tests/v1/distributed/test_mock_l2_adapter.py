# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MockL2Adapter.

Tests are written based on the L2AdapterInterface contract defined in base.py.
Tests only use public methods and do not access private fields.
"""

# Standard
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


class _RecordingListener(L2AdapterListener):
    """Listener that records all events for inspection in tests."""

    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self.deleted.append(list(keys))


# =============================================================================
# Test Fixtures
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 1024, fill_value: float = 1.0) -> TensorMemoryObj:
    """Create a test TensorMemoryObj with the given size."""
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """
    Wait for an event fd to be signaled.

    Returns True if signaled within timeout, False otherwise.
    """
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)  # timeout in milliseconds
    if events:
        # Read and consume the event
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


@pytest.fixture
def adapter():
    """Create a MockL2Adapter with reasonable defaults for testing."""
    config = MockL2AdapterConfig(
        max_size_gb=0.001,  # 1 MB
        mock_bandwidth_gb=10.0,  # 10 GB/s (fast for tests)
    )
    adapter = MockL2Adapter(config)
    yield adapter
    adapter.close()


@pytest.fixture
def slow_adapter():
    """Create a MockL2Adapter with slow bandwidth for timing tests."""
    config = MockL2AdapterConfig(
        max_size_gb=0.001,  # 1 MB
        mock_bandwidth_gb=0.0001,  # Very slow bandwidth
    )
    adapter = MockL2Adapter(config)
    yield adapter
    adapter.close()


# =============================================================================
# Event Fd Interface Tests
# =============================================================================


class TestEventFdInterface:
    """Test the event fd interface methods."""

    def test_get_store_event_fd_returns_valid_fd(self, adapter):
        """get_store_event_fd should return a valid file descriptor."""
        fd = adapter.get_store_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_lookup_and_lock_event_fd_returns_valid_fd(self, adapter):
        """get_lookup_and_lock_event_fd should return a valid file descriptor."""
        fd = adapter.get_lookup_and_lock_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_load_event_fd_returns_valid_fd(self, adapter):
        """get_load_event_fd should return a valid file descriptor."""
        fd = adapter.get_load_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_event_fds_are_different(self, adapter):
        """Each operation should have a distinct event fd."""
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        assert store_fd != lookup_fd
        assert store_fd != load_fd
        assert lookup_fd != load_fd


# =============================================================================
# Store Interface Tests
# =============================================================================


class TestStoreInterface:
    """Test the store operation interface."""

    def test_submit_store_task_returns_task_id(self, adapter):
        """submit_store_task should return a valid task ID."""
        key = create_object_key(1)
        obj = create_memory_obj()

        task_id = adapter.submit_store_task([key], [obj])

        assert isinstance(task_id, int)

    def test_submit_store_task_signals_event_fd(self, adapter):
        """Store completion should signal the store event fd."""
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        adapter.submit_store_task([key], [obj])

        # Wait for the event fd to be signaled
        assert wait_for_event_fd(store_fd, timeout=5.0), (
            "Store event fd was not signaled within timeout"
        )

    def test_pop_completed_store_tasks_returns_completed(self, adapter):
        """pop_completed_store_tasks should return completed tasks
        with success status.
        """
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task([key], [obj])

        # Wait for completion
        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adapter.pop_completed_store_tasks()

        assert task_id in completed
        assert completed[task_id].is_successful()  # Should be successful

    def test_pop_completed_store_tasks_clears_completed(self, adapter):
        """pop_completed_store_tasks should clear the completed tasks."""
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)

        # First pop should return the task
        completed1 = adapter.pop_completed_store_tasks()
        assert len(completed1) == 1

        # Second pop should return empty
        completed2 = adapter.pop_completed_store_tasks()
        assert len(completed2) == 0

    def test_submit_multiple_store_tasks(self, adapter):
        """Multiple store tasks should each get unique task IDs."""
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(3)]
        store_fd = adapter.get_store_event_fd()

        task_ids = []
        for key, obj in zip(keys, objs, strict=False):
            task_id = adapter.submit_store_task([key], [obj])
            task_ids.append(task_id)

        # All task IDs should be unique
        assert len(set(task_ids)) == 3

        # Wait for all completions
        completed = {}
        while len(completed) < 3:
            wait_for_event_fd(store_fd, timeout=5.0)
            completed.update(adapter.pop_completed_store_tasks())

        # All tasks should be completed
        for task_id in task_ids:
            assert task_id in completed
            assert completed[task_id].is_successful()

    def test_store_batch_of_objects(self, adapter):
        """A single store task can store multiple key-object pairs."""
        keys = [create_object_key(i) for i in range(5)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(5)]
        store_fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task(keys, objs)

        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adapter.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id].is_successful()


# =============================================================================
# Lookup and Lock Interface Tests
# =============================================================================


class TestLookupAndLockInterface:
    """Test the lookup and lock operation interface."""

    def test_submit_lookup_returns_task_id(self, adapter):
        """submit_lookup_and_lock_task should return a valid task ID."""
        key = create_object_key(1)

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)

        assert isinstance(task_id, int)

    def test_lookup_signals_event_fd(self, adapter):
        """Lookup completion should signal the lookup event fd."""
        key = create_object_key(1)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)

        assert wait_for_event_fd(lookup_fd, timeout=5.0), (
            "Lookup event fd was not signaled within timeout"
        )

    def test_lookup_nonexistent_key_returns_bitmap_with_zeros(self, adapter):
        """Looking up a non-existent key should return a bitmap with 0."""
        key = create_object_key(999)  # Never stored
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is False  # Key not found

    def test_lookup_existing_key_returns_bitmap_with_ones(self, adapter):
        """Looking up an existing key should return a bitmap with 1."""
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # First store the object
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Now lookup
        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is True  # Key found

    def test_lookup_mixed_keys(self, adapter):
        """Lookup of mixed existing/non-existing keys returns correct bitmap."""
        existing_key = create_object_key(1)
        nonexistent_key = create_object_key(999)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store only one key
        adapter.submit_store_task([existing_key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Lookup both keys
        task_id = adapter.submit_lookup_and_lock_task(
            [existing_key, nonexistent_key], _EMPTY_LAYOUT
        )
        wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is True  # existing_key found
        assert bitmap.test(1) is False  # nonexistent_key not found

    def test_query_lookup_result_returns_none_for_unknown_task(self, adapter):
        """Querying an unknown task ID should return None."""
        result = adapter.query_lookup_and_lock_result(99999)
        assert result is None

    def test_query_lookup_result_clears_result(self, adapter):
        """Querying lookup result should remove it (can only query once)."""
        key = create_object_key(1)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        # First query returns result
        result1 = adapter.query_lookup_and_lock_result(task_id)
        assert result1 is not None

        # Second query returns None
        result2 = adapter.query_lookup_and_lock_result(task_id)
        assert result2 is None


# =============================================================================
# Unlock Interface Tests
# =============================================================================


class TestUnlockInterface:
    """Test the unlock operation interface."""

    def test_submit_unlock_does_not_raise(self, adapter):
        """submit_unlock should not raise an exception."""
        key = create_object_key(1)

        # Should not raise even for non-existent key
        adapter.submit_unlock([key])

    def test_unlock_after_lock(self, adapter):
        """Unlocking after locking should work without error."""
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)
        adapter.query_lookup_and_lock_result(task_id)

        # Unlock should not raise
        adapter.submit_unlock([key])


# =============================================================================
# Load Interface Tests
# =============================================================================


class TestLoadInterface:
    """Test the load operation interface."""

    def test_submit_load_task_returns_task_id(self, adapter):
        """submit_load_task should return a valid task ID."""
        key = create_object_key(1)
        obj = create_memory_obj()

        task_id = adapter.submit_load_task([key], [obj])

        assert isinstance(task_id, int)

    def test_load_signals_event_fd(self, adapter):
        """Load completion should signal the load event fd."""
        key = create_object_key(1)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        adapter.submit_load_task([key], [obj])

        assert wait_for_event_fd(load_fd, timeout=5.0), (
            "Load event fd was not signaled within timeout"
        )

    def test_load_nonexistent_key_returns_bitmap_with_zeros(self, adapter):
        """Loading a non-existent key should return a bitmap with 0."""
        key = create_object_key(999)  # Never stored
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        task_id = adapter.submit_load_task([key], [obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adapter.query_load_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is False  # Load failed

    def test_load_existing_key_copies_data(self, adapter):
        """Loading an existing key should copy data to the provided buffer."""
        key = create_object_key(1)
        store_obj = create_memory_obj(size=100, fill_value=42.0)
        load_obj = create_memory_obj(size=100, fill_value=0.0)
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store
        adapter.submit_store_task([key], [store_obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Load
        task_id = adapter.submit_load_task([key], [load_obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adapter.query_load_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is True  # Load succeeded

        # Data should be copied
        assert torch.all(load_obj.tensor == 42.0)

    def test_query_load_result_returns_none_for_unknown_task(self, adapter):
        """Querying an unknown task ID should return None."""
        result = adapter.query_load_result(99999)
        assert result is None

    def test_query_load_result_clears_result(self, adapter):
        """Querying load result should remove it (can only query once)."""
        key = create_object_key(1)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        task_id = adapter.submit_load_task([key], [obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        # First query returns result
        result1 = adapter.query_load_result(task_id)
        assert result1 is not None

        # Second query returns None
        result2 = adapter.query_load_result(task_id)
        assert result2 is None


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflow:
    """Test complete store-lookup-load workflows."""

    def test_store_lookup_load_workflow(self, adapter):
        """Test the complete workflow: store -> lookup -> load."""
        key = create_object_key(1)
        store_obj = create_memory_obj(size=256, fill_value=123.0)
        load_obj = create_memory_obj(size=256, fill_value=0.0)

        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Step 1: Store
        store_task_id = adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        completed = adapter.pop_completed_store_tasks()
        assert completed[store_task_id].is_successful()

        # Step 2: Lookup and lock
        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap.test(0) is True

        # Step 3: Load
        load_task_id = adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)
        load_bitmap = adapter.query_load_result(load_task_id)
        assert load_bitmap.test(0) is True

        # Verify data
        assert torch.all(load_obj.tensor == 123.0)

        # Step 4: Unlock
        adapter.submit_unlock([key])

    def test_multiple_objects_workflow(self, adapter):
        """Test workflow with multiple objects."""
        num_objects = 5
        keys = [create_object_key(i) for i in range(num_objects)]
        store_objs = [
            create_memory_obj(size=64, fill_value=float(i * 10))
            for i in range(num_objects)
        ]
        load_objs = [
            create_memory_obj(size=64, fill_value=0.0) for _ in range(num_objects)
        ]

        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store all
        store_task_id = adapter.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)
        completed = adapter.pop_completed_store_tasks()
        assert completed[store_task_id].is_successful()

        # Lookup all
        lookup_task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        for i in range(num_objects):
            assert lookup_bitmap.test(i) is True

        # Load all
        load_task_id = adapter.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd, timeout=5.0)
        load_bitmap = adapter.query_load_result(load_task_id)
        for i in range(num_objects):
            assert load_bitmap.test(i) is True
            assert torch.all(load_objs[i].tensor == float(i * 10))


# =============================================================================
# Close Interface Tests
# =============================================================================


class TestCloseInterface:
    """Test the close operation."""

    def test_close_does_not_raise(self):
        """close() should not raise an exception."""
        config = MockL2AdapterConfig(max_size_gb=0.001, mock_bandwidth_gb=10.0)
        adapter = MockL2Adapter(config)

        # Should not raise
        adapter.close()

    def test_close_after_operations(self):
        """close() should work after store/lookup/load operations."""
        config = MockL2AdapterConfig(max_size_gb=0.001, mock_bandwidth_gb=10.0)
        adapter = MockL2Adapter(config)

        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Should not raise
        adapter.close()


# =============================================================================
# Bandwidth Simulation Tests
# =============================================================================


class TestBandwidthSimulation:
    """Test that bandwidth simulation affects completion timing."""

    def test_store_completion_is_delayed_by_bandwidth(self, slow_adapter):
        """Store completion should be delayed based on object size and bandwidth."""
        key = create_object_key(1)
        # Create a larger object to see noticeable delay
        obj = create_memory_obj(size=10000)  # ~40KB
        store_fd = slow_adapter.get_store_event_fd()

        start_time = time.time()
        slow_adapter.submit_store_task([key], [obj])

        # Should take some time due to bandwidth simulation
        assert wait_for_event_fd(store_fd, timeout=60.0)
        elapsed = time.time() - start_time

        # With very slow bandwidth, should take noticeable time
        # (exact time depends on bandwidth config)
        assert elapsed > 0.1  # At least some delay

    def test_fast_bandwidth_completes_quickly(self, adapter):
        """With fast bandwidth, completion should be quick."""
        key = create_object_key(1)
        obj = create_memory_obj(size=1000)
        store_fd = adapter.get_store_event_fd()

        start_time = time.time()
        adapter.submit_store_task([key], [obj])

        assert wait_for_event_fd(store_fd, timeout=5.0)
        elapsed = time.time() - start_time

        # With fast bandwidth, should complete quickly
        assert elapsed < 1.0


# =============================================================================
# Eviction Interface Tests
# =============================================================================


def _store_and_wait(adapter, key, obj):
    """Helper: store one key and wait for the store event fd to fire."""
    store_fd = adapter.get_store_event_fd()
    adapter.submit_store_task([key], [obj])
    assert wait_for_event_fd(store_fd, timeout=5.0), "store timed out"
    adapter.pop_completed_store_tasks()


class TestEvictionInterface:
    """Tests for delete(), get_usage(), and listener notifications."""

    def test_delete_removes_key(self, adapter):
        """delete() should make the key invisible to subsequent lookups."""
        key = create_object_key(1)
        obj = create_memory_obj()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        _store_and_wait(adapter, key, obj)

        adapter.delete([key])

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap.test(0) is False

    def test_delete_nonexistent_key_does_not_raise(self, adapter):
        """delete() on a key that was never stored should not raise."""
        adapter.delete([create_object_key(999)])

    def test_delete_multiple_keys(self, adapter):
        """delete() on a batch of keys removes all of them."""
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj() for _ in range(3)]
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        adapter.delete(keys)

        task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        for i in range(len(keys)):
            assert bitmap.test(i) is False

    def test_get_usage_empty_adapter_is_zero(self, adapter):
        """get_usage() on a fresh adapter should report 0 bytes."""
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 0
        assert usage.usage_fraction == 0.0
        assert usage.bytes_by_cache_salt == {}

    def test_get_usage_increases_after_store(self, adapter):
        """get_usage() should report positive bytes after a store."""
        key = create_object_key(1)
        obj = create_memory_obj(size=1024)
        _store_and_wait(adapter, key, obj)

        usage = adapter.get_usage()
        assert usage.total_bytes_used > 0
        assert 0.0 < usage.usage_fraction <= 1.0

    def test_get_usage_decreases_after_delete(self, adapter):
        """get_usage() should drop back to 0 after deleting the only stored key."""
        key = create_object_key(1)
        obj = create_memory_obj(size=1024)
        _store_and_wait(adapter, key, obj)

        assert adapter.get_usage().total_bytes_used > 0

        adapter.delete([key])

        usage = adapter.get_usage()
        assert usage.total_bytes_used == 0
        assert usage.usage_fraction == 0.0

    def test_bytes_by_cache_salt_populated_from_cache_salt(self, adapter):
        """End-to-end: storing keys with different ``cache_salt`` values
        should drive the byte buckets in ``AdapterUsage`` — proves the
        salt actually flows through the real adapter into the base-class
        accounting (not just verified by the stub tests)."""
        # Two keys per cache_salt so we know the totals are summed, not
        # just per-key-overwritten.
        alice_keys = [
            ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(i),
                model_name="m",
                kv_rank=0,
                cache_salt="alice",
            )
            for i in (1, 2)
        ]
        bob_key = ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(3),
            model_name="m",
            kv_rank=0,
            cache_salt="bob",
        )
        obj = create_memory_obj(size=128)  # 128 floats * 4 bytes = 512 B
        for k in alice_keys + [bob_key]:
            _store_and_wait(adapter, k, obj)

        usage = adapter.get_usage()
        assert usage.bytes_by_cache_salt == {"alice": 1024, "bob": 512}
        assert usage.total_bytes_used == 1536

        # Deleting one of alice's keys should shrink alice's bucket but
        # leave bob's untouched.
        adapter.delete([alice_keys[0]])
        usage = adapter.get_usage()
        assert usage.bytes_by_cache_salt == {"alice": 512, "bob": 512}
        assert usage.total_bytes_used == 1024

        # Deleting alice's last key should drop the bucket entirely so
        # the snapshot stays compact.
        adapter.delete([alice_keys[1]])
        usage = adapter.get_usage()
        assert "alice" not in usage.bytes_by_cache_salt
        assert usage.bytes_by_cache_salt == {"bob": 512}

    def test_listener_notified_on_store(self, adapter):
        """Listener.on_l2_keys_stored should be called after a store completes."""
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj()
        _store_and_wait(adapter, key, obj)

        assert len(listener.stored) == 1
        assert key in listener.stored[0]
        assert listener.deleted == []

    def test_listener_notified_on_delete(self, adapter):
        """Listener.on_l2_keys_deleted should be called after delete()."""
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj()
        _store_and_wait(adapter, key, obj)
        adapter.delete([key])

        assert len(listener.deleted) == 1
        assert key in listener.deleted[0]

    def test_listener_delete_skips_missing_keys(self, adapter):
        """on_l2_keys_deleted should only include keys that were actually removed."""
        listener = _RecordingListener()
        adapter.register_listener(listener)

        real_key = create_object_key(1)
        missing_key = create_object_key(999)
        obj = create_memory_obj()
        _store_and_wait(adapter, real_key, obj)

        adapter.delete([real_key, missing_key])

        assert len(listener.deleted) == 1
        notified = listener.deleted[0]
        assert real_key in notified
        assert missing_key not in notified

    def test_listener_notified_on_load(self, adapter):
        """Listener.on_l2_keys_accessed should be called after a load completes."""
        listener = _RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(1)
        store_obj = create_memory_obj(size=100, fill_value=42.0)
        load_obj = create_memory_obj(size=100, fill_value=0.0)
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store
        adapter.submit_store_task([key], [store_obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Load
        adapter.submit_load_task([key], [load_obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        assert len(listener.accessed) == 1
        assert key in listener.accessed[0]

    def test_listener_load_skips_missing_keys(self, adapter):
        """on_l2_keys_accessed should only include keys that were actually loaded."""
        listener = _RecordingListener()
        adapter.register_listener(listener)

        real_key = create_object_key(1)
        missing_key = create_object_key(999)
        store_obj = create_memory_obj(size=100, fill_value=42.0)
        load_obj1 = create_memory_obj(size=100, fill_value=0.0)
        load_obj2 = create_memory_obj(size=100, fill_value=0.0)
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store only real_key
        adapter.submit_store_task([real_key], [store_obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Load both keys
        adapter.submit_load_task([real_key, missing_key], [load_obj1, load_obj2])
        wait_for_event_fd(load_fd, timeout=5.0)

        assert len(listener.accessed) == 1
        assert real_key in listener.accessed[0]
        assert missing_key not in listener.accessed[0]

    def test_multiple_listeners_all_notified(self, adapter):
        """All registered listeners should receive the same store event."""
        l1, l2 = _RecordingListener(), _RecordingListener()
        adapter.register_listener(l1)
        adapter.register_listener(l2)

        key = create_object_key(1)
        obj = create_memory_obj()
        _store_and_wait(adapter, key, obj)

        assert len(l1.stored) == 1
        assert len(l2.stored) == 1
