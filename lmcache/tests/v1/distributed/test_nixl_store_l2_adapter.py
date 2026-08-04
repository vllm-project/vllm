# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NixlStoreL2Adapter with POSIX backend.

Tests are written based on the L2AdapterInterface contract defined in base.py.
Tests only use public methods and do not access private fields.
"""

# Standard
from unittest.mock import call, patch
import errno
import os
import select
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type

nixl = pytest.importorskip("nixl")
if torch_device_type == "xpu":
    pytest.skip(
        (
            "Skip on XPU: in vllm/vllm-openai-xpu:v0.26.0, "
            "NIXL dynamic store backends are unavailable at runtime "
            "(including POSIX), adapter init can fail with "
            "NIXL_ERR_NOT_FOUND, so this suite is not runnable "
            "on XPU in the current test environment."
        ),
        allow_module_level=True,
    )
# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.internal_api import (  # noqa: E402
    L1MemoryDesc,
    L2AdapterListener,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (  # noqa: E402
    NixlStoreL2Adapter,
    NixlStoreL2AdapterConfig,
)


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


# First Party
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# =============================================================================
# Constants
# =============================================================================

PAGE_SIZE = 4096  # 4 KB per page
NUM_BUFFER_PAGES = 20  # pages in the registered memory buffer
POOL_SIZE = 20  # number of storage descriptors to pre-allocate

# =============================================================================
# Test Helpers
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(
    buffer: torch.Tensor,
    page_index: int,
    fill_value: float = 1.0,
    num_pages: int = 1,
) -> TensorMemoryObj:
    """Create a TensorMemoryObj that references page(s) in the registered buffer.

    Args:
        buffer: The flat uint8 buffer registered with NIXL.
        page_index: Starting page in the buffer this object occupies.
        fill_value: Value to fill the tensor with.
        num_pages: Number of contiguous pages this object spans.
    """
    obj_size = PAGE_SIZE * num_pages
    start = page_index * PAGE_SIZE
    end = start + obj_size
    num_floats = obj_size // 4

    raw_data = buffer[start:end].view(torch.float32)
    raw_data.fill_(fill_value)

    metadata = MemoryObjMetadata(
        shape=torch.Size([num_floats]),
        dtype=torch.float32,
        address=page_index * PAGE_SIZE,
        phy_size=obj_size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """Wait for an event fd to be signaled.

    Returns True if signaled within timeout, False otherwise.
    """
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)  # timeout in milliseconds
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def adapter():
    """Create a NixlStoreL2Adapter with POSIX backend and an initialized
    memory buffer.

    Yields (adapter, buffer) so tests can create memory objects that
    reference pages inside the registered buffer.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nixl_l2_test_")

    # Allocate a contiguous CPU buffer first so we can pass it to the adapter
    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu")

    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(),
        size=buffer.numel(),
        align_bytes=PAGE_SIZE,
    )

    config = NixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={"file_path": tmp_dir, "use_direct_io": "false"},
        pool_size=POOL_SIZE,
    )
    adapter = NixlStoreL2Adapter(config, l1_memory)

    yield adapter, buffer

    adapter.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Event Fd Interface Tests
# =============================================================================


class TestEventFdInterface:
    """Test the event fd interface methods."""

    def test_get_store_event_fd_returns_valid_fd(self, adapter):
        """get_store_event_fd should return a valid file descriptor."""
        adpt, _ = adapter
        fd = adpt.get_store_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_lookup_and_lock_event_fd_returns_valid_fd(self, adapter):
        """get_lookup_and_lock_event_fd should return a valid file descriptor."""
        adpt, _ = adapter
        fd = adpt.get_lookup_and_lock_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_load_event_fd_returns_valid_fd(self, adapter):
        """get_load_event_fd should return a valid file descriptor."""
        adpt, _ = adapter
        fd = adpt.get_load_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_event_fds_are_different(self, adapter):
        """Each operation should have a distinct event fd."""
        adpt, _ = adapter
        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()
        load_fd = adpt.get_load_event_fd()

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
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        task_id = adpt.submit_store_task([key], [obj])

        assert isinstance(task_id, int)

    def test_submit_store_task_signals_event_fd(self, adapter):
        """Store completion should signal the store event fd."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        store_fd = adpt.get_store_event_fd()

        adpt.submit_store_task([key], [obj])

        assert wait_for_event_fd(store_fd, timeout=5.0), (
            "Store event fd was not signaled within timeout"
        )

    def test_pop_completed_store_tasks_returns_completed(self, adapter):
        """pop_completed_store_tasks should return completed tasks
        with success status.
        """
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        store_fd = adpt.get_store_event_fd()

        task_id = adpt.submit_store_task([key], [obj])

        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adpt.pop_completed_store_tasks()

        assert task_id in completed
        assert completed[task_id].is_successful()

    def test_pop_completed_store_tasks_clears_completed(self, adapter):
        """pop_completed_store_tasks should clear the completed tasks."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        store_fd = adpt.get_store_event_fd()

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)

        # First pop should return the task
        completed1 = adpt.pop_completed_store_tasks()
        assert len(completed1) == 1

        # Second pop should return empty
        completed2 = adpt.pop_completed_store_tasks()
        assert len(completed2) == 0

    def test_submit_multiple_store_tasks(self, adapter):
        """Multiple store tasks should each get unique task IDs."""
        adpt, buf = adapter
        keys = [create_object_key(i) for i in range(3)]
        objs = [
            create_memory_obj(buf, page_index=i, fill_value=float(i)) for i in range(3)
        ]
        store_fd = adpt.get_store_event_fd()

        task_ids = []
        for key, obj in zip(keys, objs, strict=False):
            task_id = adpt.submit_store_task([key], [obj])
            task_ids.append(task_id)

        # All task IDs should be unique
        assert len(set(task_ids)) == 3

        # Wait for all completions
        completed = {}
        while len(completed) < 3:
            wait_for_event_fd(store_fd, timeout=5.0)
            completed.update(adpt.pop_completed_store_tasks())

        for task_id in task_ids:
            assert task_id in completed
            assert completed[task_id].is_successful()

    def test_store_batch_of_objects(self, adapter):
        """A single store task can store multiple key-object pairs."""
        adpt, buf = adapter
        keys = [create_object_key(i) for i in range(5)]
        objs = [
            create_memory_obj(buf, page_index=i, fill_value=float(i)) for i in range(5)
        ]
        store_fd = adpt.get_store_event_fd()

        task_id = adpt.submit_store_task(keys, objs)

        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adpt.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id].is_successful()


# =============================================================================
# Lookup and Lock Interface Tests
# =============================================================================


class TestLookupAndLockInterface:
    """Test the lookup and lock operation interface."""

    def test_submit_lookup_returns_task_id(self, adapter):
        """submit_lookup_and_lock_task should return a valid task ID."""
        adpt, _ = adapter
        key = create_object_key(1)

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)

        assert isinstance(task_id, int)

    def test_lookup_signals_event_fd(self, adapter):
        """Lookup completion should signal the lookup event fd."""
        adpt, _ = adapter
        key = create_object_key(1)
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)

        assert wait_for_event_fd(lookup_fd, timeout=5.0), (
            "Lookup event fd was not signaled within timeout"
        )

    def test_lookup_nonexistent_key_returns_bitmap_with_zeros(self, adapter):
        """Looking up a non-existent key should return a bitmap with 0."""
        adpt, _ = adapter
        key = create_object_key(999)  # Never stored
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adpt.query_lookup_and_lock_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is False  # Key not found

    def test_lookup_existing_key_returns_bitmap_with_ones(self, adapter):
        """Looking up an existing key should return a bitmap with 1."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        # First store the object
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Now lookup
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adpt.query_lookup_and_lock_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is True  # Key found

    def test_lookup_mixed_keys(self, adapter):
        """Lookup of mixed existing/non-existing keys returns correct bitmap."""
        adpt, buf = adapter
        existing_key = create_object_key(1)
        nonexistent_key = create_object_key(999)
        obj = create_memory_obj(buf, page_index=0)
        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        # Store only one key
        adpt.submit_store_task([existing_key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Lookup both keys
        task_id = adpt.submit_lookup_and_lock_task(
            [existing_key, nonexistent_key], _EMPTY_LAYOUT
        )
        wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adpt.query_lookup_and_lock_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is True  # existing_key found
        assert bitmap.test(1) is False  # nonexistent_key not found

    def test_query_lookup_result_returns_none_for_unknown_task(self, adapter):
        """Querying an unknown task ID should return None."""
        adpt, _ = adapter
        result = adpt.query_lookup_and_lock_result(99999)
        assert result is None

    def test_query_lookup_result_clears_result(self, adapter):
        """Querying lookup result should remove it (can only query once)."""
        adpt, _ = adapter
        key = create_object_key(1)
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        # First query returns result
        result1 = adpt.query_lookup_and_lock_result(task_id)
        assert result1 is not None

        # Second query returns None
        result2 = adpt.query_lookup_and_lock_result(task_id)
        assert result2 is None


# =============================================================================
# Unlock Interface Tests
# =============================================================================


class TestUnlockInterface:
    """Test the unlock operation interface."""

    def test_submit_unlock_does_not_raise(self, adapter):
        """submit_unlock should not raise an exception."""
        adpt, _ = adapter
        key = create_object_key(1)

        # Should not raise even for non-existent key
        adpt.submit_unlock([key])

    def test_unlock_after_lock(self, adapter):
        """Unlocking after locking should work without error."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        # Store
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)
        adpt.query_lookup_and_lock_result(task_id)

        # Unlock should not raise
        adpt.submit_unlock([key])


# =============================================================================
# Load Interface Tests
# =============================================================================


class TestLoadInterface:
    """Test the load operation interface."""

    def test_submit_load_task_returns_task_id(self, adapter):
        """submit_load_task should return a valid task ID."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        task_id = adpt.submit_load_task([key], [obj])

        assert isinstance(task_id, int)

    def test_load_signals_event_fd(self, adapter):
        """Load completion should signal the load event fd."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        load_fd = adpt.get_load_event_fd()

        adpt.submit_load_task([key], [obj])

        assert wait_for_event_fd(load_fd, timeout=5.0), (
            "Load event fd was not signaled within timeout"
        )

    def test_load_nonexistent_key_returns_bitmap_with_zeros(self, adapter):
        """Loading a non-existent key should return a bitmap with 0."""
        adpt, buf = adapter
        key = create_object_key(999)  # Never stored
        obj = create_memory_obj(buf, page_index=0)
        load_fd = adpt.get_load_event_fd()

        task_id = adpt.submit_load_task([key], [obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adpt.query_load_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is False  # Load failed

    def test_load_existing_key_copies_data(self, adapter):
        """Loading an existing key should copy data to the provided buffer."""
        adpt, buf = adapter
        key = create_object_key(1)

        # Store from page 0 (filled with 42.0)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=42.0)
        store_fd = adpt.get_store_event_fd()
        load_fd = adpt.get_load_event_fd()

        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Load into page 1 (initially filled with 0.0)
        load_obj = create_memory_obj(buf, page_index=1, fill_value=0.0)
        task_id = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adpt.query_load_result(task_id)

        assert bitmap is not None
        assert bitmap.test(0) is True  # Load succeeded

        # Data should be copied
        assert torch.all(load_obj.raw_data == 42.0)

    def test_query_load_result_returns_none_for_unknown_task(self, adapter):
        """Querying an unknown task ID should return None."""
        adpt, _ = adapter
        result = adpt.query_load_result(99999)
        assert result is None

    def test_query_load_result_clears_result(self, adapter):
        """Querying load result should remove it (can only query once)."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        load_fd = adpt.get_load_event_fd()

        task_id = adpt.submit_load_task([key], [obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        # First query returns result
        result1 = adpt.query_load_result(task_id)
        assert result1 is not None

        # Second query returns None
        result2 = adpt.query_load_result(task_id)
        assert result2 is None


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflow:
    """Test complete store-lookup-load workflows."""

    def test_store_lookup_load_workflow(self, adapter):
        """Test the complete workflow: store -> lookup -> load."""
        adpt, buf = adapter
        key = create_object_key(1)

        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()
        load_fd = adpt.get_load_event_fd()

        # Step 1: Store from page 0 (filled with 123.0)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=123.0)
        store_task_id = adpt.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        completed = adpt.pop_completed_store_tasks()
        assert completed[store_task_id].is_successful()

        # Step 2: Lookup and lock
        lookup_task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        lookup_bitmap = adpt.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap.test(0) is True

        # Step 3: Load into page 1 (initially 0.0)
        load_obj = create_memory_obj(buf, page_index=1, fill_value=0.0)
        load_task_id = adpt.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)
        load_bitmap = adpt.query_load_result(load_task_id)
        assert load_bitmap.test(0) is True

        # Verify data
        assert torch.all(load_obj.raw_data == 123.0)

        # Step 4: Unlock
        adpt.submit_unlock([key])

    def test_multi_page_object_workflow(self, adapter):
        """Test store-lookup-load with an object spanning multiple pages."""
        adpt, buf = adapter
        key = create_object_key(1)
        num_pages = 3

        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()
        load_fd = adpt.get_load_event_fd()

        # Store a 3-page object starting at page 0
        store_obj = create_memory_obj(
            buf, page_index=0, fill_value=77.0, num_pages=num_pages
        )
        store_task_id = adpt.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        completed = adpt.pop_completed_store_tasks()
        assert completed[store_task_id].is_successful()

        # Lookup
        lookup_task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        lookup_bitmap = adpt.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap.test(0) is True

        # Load into pages 10..12 (initially 0.0)
        load_obj = create_memory_obj(
            buf, page_index=10, fill_value=0.0, num_pages=num_pages
        )
        load_task_id = adpt.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)
        load_bitmap = adpt.query_load_result(load_task_id)
        assert load_bitmap.test(0) is True

        # All 3 pages should contain 77.0
        assert torch.all(load_obj.raw_data == 77.0)

        adpt.submit_unlock([key])

    def test_multiple_objects_workflow(self, adapter):
        """Test workflow with multiple objects."""
        adpt, buf = adapter
        num_objects = 5

        keys = [create_object_key(i) for i in range(num_objects)]

        store_fd = adpt.get_store_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()
        load_fd = adpt.get_load_event_fd()

        # Store all (pages 0..4, each filled with i * 10.0)
        store_objs = [
            create_memory_obj(buf, page_index=i, fill_value=float(i * 10))
            for i in range(num_objects)
        ]
        store_task_id = adpt.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)
        completed = adpt.pop_completed_store_tasks()
        assert completed[store_task_id].is_successful()

        # Lookup all
        lookup_task_id = adpt.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        lookup_bitmap = adpt.query_lookup_and_lock_result(lookup_task_id)
        for i in range(num_objects):
            assert lookup_bitmap.test(i) is True

        # Load all (pages 10..14, initially 0.0)
        load_objs = [
            create_memory_obj(buf, page_index=10 + i, fill_value=0.0)
            for i in range(num_objects)
        ]
        load_task_id = adpt.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd, timeout=5.0)
        load_bitmap = adpt.query_load_result(load_task_id)
        for i in range(num_objects):
            assert load_bitmap.test(i) is True
            assert torch.all(load_objs[i].raw_data == float(i * 10))


# =============================================================================
# Close Interface Tests
# =============================================================================


class TestCloseInterface:
    """Test the close operation."""

    def test_close_releases_nixl_resources(self, tmp_path):
        """close() should release NIXL handles, registrations, and storage FDs."""
        buffer = torch.empty(
            PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
        )
        l1_memory = L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=buffer.numel(),
            align_bytes=PAGE_SIZE,
        )
        config = NixlStoreL2AdapterConfig(
            backend="POSIX",
            backend_params={
                "file_path": str(tmp_path),
                "use_direct_io": "false",
            },
            pool_size=POOL_SIZE,
        )
        adpt = NixlStoreL2Adapter(config, l1_memory)
        storage_agent = adpt.nixl_agent
        storage_fds = list(storage_agent.storage_fds)

        with (
            patch.object(
                storage_agent.nixl_agent,
                "release_dlist_handle",
                wraps=storage_agent.nixl_agent.release_dlist_handle,
            ) as release_dlist_handle,
            patch.object(
                storage_agent.nixl_agent,
                "deregister_memory",
                wraps=storage_agent.nixl_agent.deregister_memory,
            ) as deregister_memory,
        ):
            adpt.close()

        assert release_dlist_handle.call_args_list == [
            call(storage_agent.storage_xfer_handler),
            call(storage_agent.mem_xfer_handler),
        ]
        assert deregister_memory.call_args_list == [
            call(storage_agent.storage_reg_descs),
            call(storage_agent.mem_reg_descs),
        ]
        for fd in storage_fds:
            with pytest.raises(OSError) as exc_info:
                os.fstat(fd)
            assert exc_info.value.errno == errno.EBADF

    def test_close_does_not_raise(self):
        """close() should not raise an exception."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_l2_close_test_")
        buffer = torch.empty(
            PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
        )
        l1_memory = L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=buffer.numel(),
            align_bytes=PAGE_SIZE,
        )
        config = NixlStoreL2AdapterConfig(
            backend="POSIX",
            backend_params={"file_path": tmp_dir, "use_direct_io": "false"},
            pool_size=POOL_SIZE,
        )
        adpt = NixlStoreL2Adapter(config, l1_memory)

        # Should not raise
        adpt.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_close_after_operations(self):
        """close() should work after store/lookup/load operations."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_l2_close_ops_test_")
        buffer = torch.empty(
            PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
        )
        l1_memory = L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=buffer.numel(),
            align_bytes=PAGE_SIZE,
        )
        config = NixlStoreL2AdapterConfig(
            backend="POSIX",
            backend_params={"file_path": tmp_dir, "use_direct_io": "false"},
            pool_size=POOL_SIZE,
        )
        adpt = NixlStoreL2Adapter(config, l1_memory)

        key = create_object_key(1)
        obj = create_memory_obj(buffer, page_index=0)
        store_fd = adpt.get_store_event_fd()

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Should not raise
        adpt.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Report Status Tests
# =============================================================================


class TestReportStatus:
    """Tests for NixlStoreL2Adapter.report_status()."""

    def test_report_status_shape(self, adapter):
        """report_status() should return all expected keys with correct types."""
        adpt, _ = adapter
        status = adpt.report_status()

        assert status["is_healthy"] is True
        assert status["type"] == "NixlStoreL2Adapter"
        assert status["backend"] == "POSIX"
        assert isinstance(status["stored_object_count"], int)
        assert isinstance(status["pinned_object_count"], int)
        assert isinstance(status["pool_size"], int)
        assert isinstance(status["pool_free_slots"], int)
        assert isinstance(status["event_loop_alive"], bool)

    def test_report_status_initial_state(self, adapter):
        """Fresh adapter should report zero stored objects and full pool."""
        adpt, _ = adapter
        status = adpt.report_status()

        assert status["stored_object_count"] == 0
        assert status["pinned_object_count"] == 0
        assert status["pool_size"] == POOL_SIZE
        assert status["pool_free_slots"] > 0
        assert status["event_loop_alive"] is True

    def test_report_status_after_store(self, adapter):
        """stored_object_count should increase after a store completes."""
        adpt, buffer = adapter
        store_fd = adpt.get_store_event_fd()

        key = create_object_key(42)
        obj = create_memory_obj(buffer, page_index=0)
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        status = adpt.report_status()
        assert status["stored_object_count"] == 1
        assert status["pool_free_slots"] < status["pool_size"]

    def test_report_status_after_close(self):
        """is_healthy should become False after close()."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_l2_status_test_")
        buffer = torch.empty(
            PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
        )
        l1_memory = L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=buffer.numel(),
            align_bytes=PAGE_SIZE,
        )
        config = NixlStoreL2AdapterConfig(
            backend="POSIX",
            backend_params={"file_path": tmp_dir, "use_direct_io": "false"},
            pool_size=POOL_SIZE,
        )
        adpt = NixlStoreL2Adapter(config, l1_memory)

        # Healthy before close
        assert adpt.report_status()["is_healthy"] is True

        adpt.close()

        # Unhealthy after close
        status = adpt.report_status()
        assert status["is_healthy"] is False
        assert status["event_loop_alive"] is False

        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Eviction Interface Tests
# =============================================================================


def _store_and_wait(adpt, key, obj):
    """Helper: store one key and wait for the store event fd to fire."""
    store_fd = adpt.get_store_event_fd()
    adpt.submit_store_task([key], [obj])
    assert wait_for_event_fd(store_fd, timeout=5.0), "store timed out"
    adpt.pop_completed_store_tasks()


class TestEvictionInterface:
    """Tests for delete(), get_usage(), and listener notifications."""

    def test_delete_removes_key(self, adapter):
        """delete() should make the key invisible to subsequent lookups."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        _store_and_wait(adpt, key, obj)

        adpt.delete([key])

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap.test(0) is False

    def test_delete_nonexistent_key_does_not_raise(self, adapter):
        """delete() on a key that was never stored should not raise."""
        adpt, _ = adapter
        adpt.delete([create_object_key(999)])

    def test_delete_frees_pool_slot(self, adapter):
        """delete() should release the storage pool slot back to the pool."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        _store_and_wait(adpt, key, obj)
        free_slots_after_store = adpt.report_status()["pool_free_slots"]

        adpt.delete([key])

        free_slots_after_delete = adpt.report_status()["pool_free_slots"]
        assert free_slots_after_delete == free_slots_after_store + 1

    def test_get_usage_empty_adapter_is_zero(self, adapter):
        """get_usage() on a fresh adapter should report no used bytes."""
        adpt, _ = adapter
        usage = adpt.get_usage()
        assert usage.total_bytes_used == 0
        assert usage.usage_fraction == 0.0

    def test_get_usage_increases_after_store(self, adapter):
        """get_usage() should report positive utilization after a store."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        _store_and_wait(adpt, key, obj)

        usage = adpt.get_usage()
        assert usage.usage_fraction > 0.0
        assert usage.usage_fraction <= 1.0

    def test_get_usage_reflects_multiple_stores(self, adapter):
        """get_usage() should increase monotonically as more objects are stored."""
        adpt, buf = adapter
        store_fd = adpt.get_store_event_fd()

        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(buf, page_index=i) for i in range(3)]
        adpt.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        usage = adpt.get_usage()
        # 3 out of POOL_SIZE slots used
        assert usage.usage_fraction == pytest.approx(3 / POOL_SIZE)

    def test_delete_pinned_key_is_skipped(self, adapter):
        """delete() should skip a key that is pinned by an in-flight lookup."""
        adpt, buf = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        _store_and_wait(adpt, key, obj)

        # Pin the key via lookup_and_lock
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        adpt.query_lookup_and_lock_result(task_id)

        # delete should skip the pinned key — stored_object_count stays 1
        adpt.delete([key])
        assert adpt.report_status()["stored_object_count"] == 1

        # Unpin, then delete should succeed
        adpt.submit_unlock([key])
        time.sleep(0.1)  # let the unlock execute in the event loop
        adpt.delete([key])
        assert adpt.report_status()["stored_object_count"] == 0

    def test_listener_notified_on_store(self, adapter):
        """Listener.on_l2_keys_stored should be called after a store completes."""
        adpt, buf = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        _store_and_wait(adpt, key, obj)

        assert len(listener.stored) == 1
        assert key in listener.stored[0]
        assert listener.deleted == []

    def test_listener_notified_on_load(self, adapter):
        """Listener.on_l2_keys_accessed should be called after a load completes."""
        adpt, buf = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=42.0)
        store_fd = adpt.get_store_event_fd()
        load_fd = adpt.get_load_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        # Store
        adpt.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Lookup and lock (required before load)
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        adpt.query_lookup_and_lock_result(task_id)

        # Load
        load_obj = create_memory_obj(buf, page_index=1, fill_value=0.0)
        adpt.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)

        assert len(listener.accessed) == 1
        assert key in listener.accessed[0]

        adpt.submit_unlock([key])

    def test_listener_load_skips_missing_keys(self, adapter):
        """on_l2_keys_accessed should only include keys that were actually loaded."""
        adpt, buf = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        real_key = create_object_key(1)
        missing_key = create_object_key(999)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=42.0)
        store_fd = adpt.get_store_event_fd()
        load_fd = adpt.get_load_event_fd()
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        # Store only real_key
        adpt.submit_store_task([real_key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        adpt.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adpt.submit_lookup_and_lock_task([real_key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        adpt.query_lookup_and_lock_result(task_id)

        # Load both keys
        load_obj1 = create_memory_obj(buf, page_index=1, fill_value=0.0)
        load_obj2 = create_memory_obj(buf, page_index=2, fill_value=0.0)
        adpt.submit_load_task([real_key, missing_key], [load_obj1, load_obj2])
        assert wait_for_event_fd(load_fd, timeout=5.0)

        assert len(listener.accessed) == 1
        assert real_key in listener.accessed[0]
        assert missing_key not in listener.accessed[0]

        adpt.submit_unlock([real_key])

    def test_listener_notified_on_delete(self, adapter):
        """Listener.on_l2_keys_deleted should be called after delete()."""
        adpt, buf = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        _store_and_wait(adpt, key, obj)
        adpt.delete([key])

        assert len(listener.deleted) == 1
        assert key in listener.deleted[0]

    def test_listener_delete_skips_pinned_key(self, adapter):
        """
        on_l2_keys_deleted should not include keys
        that were skipped due to pinning.
        """
        adpt, buf = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)
        lookup_fd = adpt.get_lookup_and_lock_event_fd()

        _store_and_wait(adpt, key, obj)

        # Pin via lookup
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        adpt.query_lookup_and_lock_result(task_id)

        # delete while pinned — should be skipped
        adpt.delete([key])

        assert listener.deleted == []
