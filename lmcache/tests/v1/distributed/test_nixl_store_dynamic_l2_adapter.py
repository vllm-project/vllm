# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for DynamicNixlStoreL2Adapter with POSIX backend.

Tests cover the L2AdapterInterface contract, dynamic file operations,
persist, secondary lookup, and capacity management.
"""

# Standard
import os
import select
import shutil
import tempfile

# Third Party
import pytest
import torch

nixl = pytest.importorskip("nixl")

# First Party
from lmcache import torch_device_type  # noqa: E402
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.internal_api import (  # noqa: E402
    L1MemoryDesc,
    L2AdapterListener,
)
from lmcache.v1.distributed.l2_adapters.config import PersistConfig  # noqa: E402
from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (  # noqa: E402
    DynamicNixlStoreL2Adapter,
    DynamicNixlStoreL2AdapterConfig,
    _object_key_to_filename,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

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
# Constants
# =============================================================================

PAGE_SIZE = 4096  # 4 KB per page
NUM_BUFFER_PAGES = 20  # pages in the registered memory buffer
MAX_CAPACITY_GB = 0.001  # ~1 MB

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
    """Create a TensorMemoryObj that references page(s) in the registered buffer."""
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
    """Wait for an event fd to be signaled."""
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
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
    """Create a DynamicNixlStoreL2Adapter with POSIX backend.

    Yields (adapter, buffer) so tests can create memory objects that
    reference pages inside the registered buffer.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_test_")

    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu")

    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(),
        size=buffer.numel(),
        align_bytes=PAGE_SIZE,
    )

    config = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp_dir,
            "use_direct_io": "false",
            "max_capacity_gb": str(MAX_CAPACITY_GB),
        },
    )
    adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

    yield adpt, buffer, tmp_dir

    adpt.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def adapter_with_persist():
    """Create a DynamicNixlStoreL2Adapter with persist enabled.

    Yields (adapter, buffer, tmp_dir, l1_memory, config) and does NOT call
    close() — tests manage the lifecycle themselves.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_persist_test_")

    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu")

    l1_memory = L1MemoryDesc(
        ptr=buffer.data_ptr(),
        size=buffer.numel(),
        align_bytes=PAGE_SIZE,
    )

    config = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp_dir,
            "use_direct_io": "false",
            "max_capacity_gb": str(MAX_CAPACITY_GB),
        },
    )
    config.persist_config = PersistConfig(persist_enabled=True)
    adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

    yield adpt, buffer, tmp_dir, l1_memory, config

    shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Event Fd Interface Tests
# =============================================================================


class TestEventFdInterface:
    def test_get_store_event_fd_returns_valid_fd(self, adapter):
        adpt, _, _ = adapter
        fd = adpt.get_store_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_lookup_and_lock_event_fd_returns_valid_fd(self, adapter):
        adpt, _, _ = adapter
        fd = adpt.get_lookup_and_lock_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_get_load_event_fd_returns_valid_fd(self, adapter):
        adpt, _, _ = adapter
        fd = adpt.get_load_event_fd()
        assert isinstance(fd, int)
        assert fd >= 0

    def test_event_fds_are_different(self, adapter):
        adpt, _, _ = adapter
        fds = {
            adpt.get_store_event_fd(),
            adpt.get_lookup_and_lock_event_fd(),
            adpt.get_load_event_fd(),
        }
        assert len(fds) == 3


# =============================================================================
# Store Interface Tests
# =============================================================================


class TestStoreInterface:
    def test_submit_store_task_returns_task_id(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        task_id = adpt.submit_store_task([key], [obj])
        assert isinstance(task_id, int)

    def test_submit_store_task_signals_event_fd(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        assert wait_for_event_fd(adpt.get_store_event_fd())

    def test_pop_completed_store_tasks_returns_completed(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        task_id = adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())

        completed = adpt.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id].is_successful()

    def test_store_creates_file_on_disk(self, adapter):
        adpt, buf, tmp_dir = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())

        expected_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert os.path.exists(expected_file)

    def test_submit_multiple_store_tasks_unique_ids(self, adapter):
        adpt, buf, _ = adapter
        key1 = create_object_key(1)
        key2 = create_object_key(2)
        obj1 = create_memory_obj(buf, page_index=0)
        obj2 = create_memory_obj(buf, page_index=1)

        task_id1 = adpt.submit_store_task([key1], [obj1])
        task_id2 = adpt.submit_store_task([key2], [obj2])
        assert task_id1 != task_id2


# =============================================================================
# Lookup and Lock Interface Tests
# =============================================================================


class TestLookupAndLockInterface:
    def test_lookup_nonexistent_key_returns_zeros(self, adapter):
        adpt, _, _ = adapter
        key = create_object_key(999)

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())

        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert not bitmap.test(0)

    def test_lookup_existing_key_returns_ones(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        # Store first
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lookup
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())

        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        # Unlock
        adpt.submit_unlock([key])

    def test_query_lookup_result_clears_result(self, adapter):
        adpt, _, _ = adapter
        key = create_object_key(1)

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())

        result1 = adpt.query_lookup_and_lock_result(task_id)
        result2 = adpt.query_lookup_and_lock_result(task_id)
        assert result1 is not None
        assert result2 is None


# =============================================================================
# Load Interface Tests
# =============================================================================


class TestLoadInterface:
    def test_load_existing_key_copies_data(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=42.0)

        # Store
        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Load into a different page
        load_obj = create_memory_obj(buf, page_index=1, fill_value=0.0)
        task_id = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())

        bitmap = adpt.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        # Verify data was copied
        loaded_data = buf[PAGE_SIZE : 2 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded_data == 42.0)

        adpt.submit_unlock([key])

    def test_load_multiple_keys_concurrent(self, adapter):
        """A multi-key load task reads every chunk correctly.

        Exercises the concurrent (gather-based) load loop: store several
        keys, then load them all in a single task and verify each landed in
        its own page with the right data.
        """
        adpt, buf, _ = adapter
        keys = [create_object_key(i) for i in range(1, 4)]
        fills = [11.0, 22.0, 33.0]
        store_objs = [
            create_memory_obj(buf, page_index=i, fill_value=fills[i]) for i in range(3)
        ]

        # Store all three
        adpt.submit_store_task(keys, store_objs)
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lookup and lock
        task_id = adpt.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Load all three into separate pages (3, 4, 5) in ONE task
        load_objs = [
            create_memory_obj(buf, page_index=3 + i, fill_value=0.0) for i in range(3)
        ]
        task_id = adpt.submit_load_task(keys, load_objs)
        wait_for_event_fd(adpt.get_load_event_fd())

        bitmap = adpt.query_load_result(task_id)
        assert bitmap is not None
        for i in range(3):
            assert bitmap.test(i)
            page = 3 + i
            loaded = buf[page * PAGE_SIZE : (page + 1) * PAGE_SIZE].view(torch.float32)
            assert torch.all(loaded == fills[i])

        adpt.submit_unlock(keys)

    def test_load_partial_failure_marks_only_successful(self, adapter):
        """One chunk's failure must not discard the others in the same task.

        With ``asyncio.gather(..., return_exceptions=True)``, a single failed
        load (here: its backing file is removed) is reported as failed for
        that key while the sibling keys still load successfully.
        """
        adpt, buf, tmp_dir = adapter
        keys = [create_object_key(i) for i in range(1, 4)]
        fills = [11.0, 22.0, 33.0]
        store_objs = [
            create_memory_obj(buf, page_index=i, fill_value=fills[i]) for i in range(3)
        ]

        adpt.submit_store_task(keys, store_objs)
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        task_id = adpt.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Sabotage the middle key's backing file so its load raises.
        os.remove(os.path.join(tmp_dir, _object_key_to_filename(keys[1])))

        load_objs = [
            create_memory_obj(buf, page_index=3 + i, fill_value=0.0) for i in range(3)
        ]
        task_id = adpt.submit_load_task(keys, load_objs)
        wait_for_event_fd(adpt.get_load_event_fd())

        bitmap = adpt.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)  # loaded ok
        assert not bitmap.test(1)  # file removed -> failed, not crashed
        assert bitmap.test(2)  # loaded ok despite sibling failure

        # The two successful loads landed correct data.
        for i in (0, 2):
            page = 3 + i
            loaded = buf[page * PAGE_SIZE : (page + 1) * PAGE_SIZE].view(torch.float32)
            assert torch.all(loaded == fills[i])

        adpt.submit_unlock(keys)

    def test_query_load_result_clears_result(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=1.0)

        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        load_obj = create_memory_obj(buf, page_index=1)
        task_id = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())

        result1 = adpt.query_load_result(task_id)
        result2 = adpt.query_load_result(task_id)
        assert result1 is not None
        assert result2 is None

        adpt.submit_unlock([key])


# =============================================================================
# Store-Lookup-Load End-to-End Test
# =============================================================================


class TestEndToEnd:
    def test_store_lookup_load_workflow(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=99.0)

        # Store
        store_task = adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        completed = adpt.pop_completed_store_tasks()
        assert completed[store_task].is_successful()

        # Lookup
        lookup_task = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        bitmap = adpt.query_lookup_and_lock_result(lookup_task)
        assert bitmap is not None
        assert bitmap.test(0)

        # Load into different page
        load_obj = create_memory_obj(buf, page_index=2, fill_value=0.0)
        load_task = adpt.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt.get_load_event_fd())
        bitmap = adpt.query_load_result(load_task)
        assert bitmap is not None
        assert bitmap.test(0)

        # Verify
        loaded = buf[2 * PAGE_SIZE : 3 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded == 99.0)

        # Unlock
        adpt.submit_unlock([key])


# =============================================================================
# Eviction / Delete Interface Tests
# =============================================================================


class TestEvictionInterface:
    def test_delete_removes_key(self, adapter):
        adpt, buf, tmp_dir = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Delete
        adpt.delete([key])

        # Lookup should miss
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert not bitmap.test(0)

        # File should be removed from disk
        expected_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert not os.path.exists(expected_file)

    def test_delete_skips_pinned_key(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        # Lock
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        adpt.query_lookup_and_lock_result(task_id)

        # Delete should skip pinned key
        adpt.delete([key])

        # Should still be found
        task_id = adpt.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
        bitmap = adpt.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        adpt.submit_unlock([key])
        adpt.submit_unlock([key])

    def test_listener_notified_on_store(self, adapter):
        adpt, buf, _ = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        assert len(listener.stored) == 1
        assert key in listener.stored[0]

    def test_listener_notified_on_delete(self, adapter):
        adpt, buf, _ = adapter
        listener = _RecordingListener()
        adpt.register_listener(listener)

        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        adpt.delete([key])

        assert len(listener.deleted) == 1
        assert key in listener.deleted[0]


# =============================================================================
# Capacity / Usage Tests
# =============================================================================


class TestCapacity:
    def test_get_usage_empty_is_zero(self, adapter):
        adpt, _, _ = adapter
        usage = adpt.get_usage()
        assert usage.usage_fraction == 0.0

    def test_get_usage_increases_after_store(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        usage = adpt.get_usage()
        assert usage.usage_fraction > 0.0

    def test_get_usage_decreases_after_delete(self, adapter):
        adpt, buf, _ = adapter
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        usage_before = adpt.get_usage().usage_fraction
        adpt.delete([key])
        usage_after = adpt.get_usage().usage_fraction

        assert usage_after < usage_before

    def test_store_rejected_when_capacity_exceeded(self):
        """Store should stop when max capacity is reached."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_cap_test_")
        try:
            buffer = torch.empty(
                PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
            )
            l1_memory = L1MemoryDesc(
                ptr=buffer.data_ptr(),
                size=buffer.numel(),
                align_bytes=PAGE_SIZE,
            )
            # Very small capacity: 1 page worth of data
            tiny_cap_gb = PAGE_SIZE / (1024**3)
            config = DynamicNixlStoreL2AdapterConfig(
                backend="POSIX",
                backend_params={
                    "file_path": tmp_dir,
                    "use_direct_io": "false",
                    "max_capacity_gb": str(tiny_cap_gb),
                },
            )
            adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

            # Store first object (should succeed)
            key1 = create_object_key(1)
            obj1 = create_memory_obj(buffer, page_index=0)
            adpt.submit_store_task([key1], [obj1])
            wait_for_event_fd(adpt.get_store_event_fd())

            # Store second object (should be rejected due to capacity)
            key2 = create_object_key(2)
            obj2 = create_memory_obj(buffer, page_index=1)
            adpt.submit_store_task([key2], [obj2])
            wait_for_event_fd(adpt.get_store_event_fd())

            # Only first key should be found
            task_id = adpt.submit_lookup_and_lock_task([key1, key2], _EMPTY_LAYOUT)
            wait_for_event_fd(adpt.get_lookup_and_lock_event_fd())
            bitmap = adpt.query_lookup_and_lock_result(task_id)
            assert bitmap is not None
            assert bitmap.test(0)  # key1 found
            assert not bitmap.test(1)  # key2 not found

            adpt.submit_unlock([key1])
            adpt.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Persist / Secondary Lookup Tests
# =============================================================================


class TestPersistAndSecondaryLookup:
    def test_persist_keeps_files_on_close(self, adapter_with_persist):
        """With persist_enabled=True, data files remain on disk after close."""
        adpt, buf, tmp_dir, _, _ = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        data_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        assert os.path.exists(data_file)

        adpt.close()

        assert os.path.exists(data_file)

    def test_secondary_lookup_finds_key(self, adapter_with_persist):
        """Lookup finds keys whose files exist on disk via secondary lookup."""
        adpt, buf, _, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0, fill_value=77.0)

        # Store and close (files are kept)
        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()
        adpt.close()

        # New adapter — secondary lookup discovers the persisted file
        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)

        # Lookup should find the key via secondary lookup
        task_id = adpt2.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        bitmap = adpt2.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        adpt2.submit_unlock([key])
        adpt2.close()

    def test_secondary_lookup_and_load_data(self, adapter_with_persist):
        """After secondary lookup, load returns the same data that was stored."""
        adpt, buf, _, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        store_obj = create_memory_obj(buf, page_index=0, fill_value=55.0)

        adpt.submit_store_task([key], [store_obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()
        adpt.close()

        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)

        # Lookup (lazy recover) + load
        task_id = adpt2.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        adpt2.query_lookup_and_lock_result(task_id)

        load_obj = create_memory_obj(buf, page_index=2, fill_value=0.0)
        task_id = adpt2.submit_load_task([key], [load_obj])
        wait_for_event_fd(adpt2.get_load_event_fd())
        bitmap = adpt2.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0)

        loaded = buf[2 * PAGE_SIZE : 3 * PAGE_SIZE].view(torch.float32)
        assert torch.all(loaded == 55.0)

        adpt2.submit_unlock([key])
        adpt2.close()

    def test_secondary_lookup_misses_when_file_deleted(self, adapter_with_persist):
        """Secondary lookup returns miss for keys whose files are absent on disk."""
        adpt, buf, tmp_dir, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()
        adpt.close()

        # Delete the data file manually
        data_file = os.path.join(tmp_dir, _object_key_to_filename(key))
        os.unlink(data_file)

        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)

        task_id = adpt2.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        bitmap = adpt2.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert not bitmap.test(0)

        adpt2.close()

    def test_secondary_lookup_usage_updates(self, adapter_with_persist):
        """Secondary lookup populates _total_bytes so get_usage reflects disk files."""
        adpt, buf, _, l1_memory, config = adapter_with_persist
        key = create_object_key(1)
        obj = create_memory_obj(buf, page_index=0)

        adpt.submit_store_task([key], [obj])
        wait_for_event_fd(adpt.get_store_event_fd())
        adpt.pop_completed_store_tasks()

        usage_before = adpt.get_usage().usage_fraction
        adpt.close()

        # Right after init, usage is zero (no eager recovery)
        adpt2 = DynamicNixlStoreL2Adapter(config, l1_memory)
        usage_initial = adpt2.get_usage().usage_fraction
        assert usage_initial == 0.0

        # After a lookup, the key is populated and usage matches
        task_id = adpt2.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(adpt2.get_lookup_and_lock_event_fd())
        adpt2.query_lookup_and_lock_result(task_id)

        usage_after = adpt2.get_usage().usage_fraction
        assert usage_after == pytest.approx(usage_before, rel=1e-6)

        adpt2.submit_unlock([key])
        adpt2.close()

    def test_close_without_persist_deletes_files(self):
        """With persist_enabled=False, close() deletes all data files."""
        tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_cleanup_test_")
        try:
            buffer = torch.empty(
                PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8, device="cpu"
            )
            l1_memory = L1MemoryDesc(
                ptr=buffer.data_ptr(),
                size=buffer.numel(),
                align_bytes=PAGE_SIZE,
            )
            config = DynamicNixlStoreL2AdapterConfig(
                backend="POSIX",
                backend_params={
                    "file_path": tmp_dir,
                    "use_direct_io": "false",
                    "max_capacity_gb": str(MAX_CAPACITY_GB),
                },
            )
            config.persist_config = PersistConfig(persist_enabled=False)
            adpt = DynamicNixlStoreL2Adapter(config, l1_memory)

            key = create_object_key(1)
            obj = create_memory_obj(buffer, page_index=0)
            adpt.submit_store_task([key], [obj])
            wait_for_event_fd(adpt.get_store_event_fd())
            adpt.pop_completed_store_tasks()

            data_file = os.path.join(tmp_dir, _object_key_to_filename(key))
            assert os.path.exists(data_file)

            adpt.close()

            assert not os.path.exists(data_file)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
