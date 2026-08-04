# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NativeConnectorL2Adapter.

Uses a mock native connector (pure Python) that simulates the pybind-wrapped
C++ IStorageConnector interface, so no Redis or C++ build is needed.
"""

# Standard
import ctypes
import select
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    _object_key_to_string,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd, create_event_notifier

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# =============================================================================
# Mock Native Connector (simulates the pybind C++ IStorageConnector interface)
# =============================================================================


class MockNativeConnector:
    """
    Pure-Python mock that implements the same interface as a pybind-wrapped
    C++ IStorageConnector.  Stores data in-memory dicts.

    Methods:
      - event_fd() -> int
      - submit_batch_get(keys, memoryviews) -> int
      - submit_batch_set(keys, memoryviews) -> int
      - submit_batch_exists(keys) -> int
      - drain_completions() -> list[tuple[int, bool, str, list[bool] | None]]
      - close()
    """

    def __init__(self):
        self._efd = create_event_notifier()
        self._store: dict[str, bytes] = {}
        self._next_id = 1
        self._completions: list[tuple[int, bool, str, list[bool] | None]] = []
        self._lock = threading.Lock()
        self._closed = False

    def event_fd(self) -> int:
        return self._efd.fileno()

    def submit_batch_set(self, keys: list[str], memoryviews: list) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        try:
            for key, mv in zip(keys, memoryviews, strict=False):
                self._store[key] = bytes(mv)
            self._push_completion(fid, True, "", None)
        except Exception as e:
            self._push_completion(fid, False, str(e), None)

        return fid

    def submit_batch_get(self, keys: list[str], memoryviews: list) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        try:
            all_ok = True
            for key, mv in zip(keys, memoryviews, strict=False):
                data = self._store.get(key)
                if data is None:
                    all_ok = False
                    break
                if len(data) != mv.nbytes:
                    all_ok = False
                    break
                # Copy data into the buffer using ctypes (same as C++ void* write)
                dest_ptr = ctypes.c_char_p(
                    ctypes.addressof(ctypes.c_char.from_buffer(mv))
                )
                ctypes.memmove(dest_ptr, data, len(data))
            self._push_completion(fid, all_ok, "", None)
        except Exception as e:
            self._push_completion(fid, False, str(e), None)

        return fid

    def submit_batch_exists(self, keys: list[str]) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        results = [key in self._store for key in keys]
        self._push_completion(fid, True, "", results)

        return fid

    def submit_batch_delete(self, keys: list[str]) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1

        results = []
        for key in keys:
            if key in self._store:
                del self._store[key]
                results.append(True)
            else:
                results.append(False)
        self._push_completion(fid, True, "", results)

        return fid

    def drain_completions(self) -> list[tuple[int, bool, str, list[bool] | None]]:
        # Drain the eventfd
        try:
            self._efd.consume()
        except BlockingIOError:
            pass

        with self._lock:
            completions = list(self._completions)
            self._completions.clear()
        return completions

    def close(self):
        if not self._closed:
            self._closed = True
            self._efd.close()

    def _push_completion(
        self, fid: int, ok: bool, error: str, result_bools: list[bool] | None
    ):
        with self._lock:
            self._completions.append((fid, ok, error, result_bools))
        # Signal the eventfd
        try:
            self._efd.notify()
        except OSError:
            pass


# =============================================================================
# Test Fixtures
# =============================================================================


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 1024, fill_value: float = 1.0) -> TensorMemoryObj:
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


@pytest.fixture
def adapter():
    mock_client = MockNativeConnector()
    adp = NativeConnectorL2Adapter(mock_client)
    yield adp
    adp.close()


# =============================================================================
# ObjectKey Serialization Tests
# =============================================================================


class TestObjectKeySerialization:
    def test_serialization_is_deterministic(self):
        key = create_object_key(42, "my_model")
        s1 = _object_key_to_string(key)
        s2 = _object_key_to_string(key)
        assert s1 == s2

    def test_different_keys_produce_different_strings(self):
        k1 = create_object_key(1)
        k2 = create_object_key(2)
        assert _object_key_to_string(k1) != _object_key_to_string(k2)

    def test_serialization_format(self):
        """Unsalted keys use the 4-field shape with object_group_id
        embedded right after kv_rank."""
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
        )
        s = _object_key_to_string(key)
        assert s == "llama@000000ff@0@00010203"

    def test_object_group_id_embedded(self):
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            object_group_id=5,
        )
        s = _object_key_to_string(key)
        assert s == "llama@000000ff@5@00010203"

    def test_salted_serialization_format(self):
        """Salted keys append ``@<cache_salt>`` as a 5th field."""
        key = ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=255,
            cache_salt="alice",
        )
        s = _object_key_to_string(key)
        assert s == "llama@000000ff@0@00010203@alice"

    def test_different_salts_produce_different_strings(self):
        base = {
            "chunk_hash": b"\x00\x01\x02\x03",
            "model_name": "llama",
            "kv_rank": 0,
        }
        k_empty = ObjectKey(**base)
        k_alice = ObjectKey(**base, cache_salt="alice")
        k_bob = ObjectKey(**base, cache_salt="bob")
        s_empty = _object_key_to_string(k_empty)
        s_alice = _object_key_to_string(k_alice)
        s_bob = _object_key_to_string(k_bob)
        assert s_empty != s_alice
        assert s_alice != s_bob
        # Empty salt has no trailing "@salt", salted keys do.
        assert s_empty.count("@") == 3  # 4 fields (model, kv_rank, group, hash)
        assert s_alice.endswith("@alice")
        assert s_bob.endswith("@bob")


class TestObjectKeyModelNameValidation:
    """model_name must not contain ``@`` — the L2 adapters split keys
    and filenames on ``@`` and rely on this invariant."""

    def test_reject_at_in_model_name(self):
        with pytest.raises(ValueError, match="model_name"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="ns@model",
                kv_rank=0,
            )

    def test_slash_in_model_name_is_accepted(self):
        # '/' is sanitized to '-SEP-' by the FS adapter; the invariant
        # is only about '@'.
        key = ObjectKey(
            chunk_hash=b"\x00",
            model_name="meta-llama/Llama-3",
            kv_rank=0,
        )
        assert key.model_name == "meta-llama/Llama-3"


class TestObjectKeyCacheSaltValidation:
    """cache_salt must not contain ``@``, ``/``, ``\\``, or NUL, and must
    be <= 128 chars. The invariant is enforced at construction time so
    all downstream serializers (Python + C++) can rely on it."""

    def test_reject_at_in_salt(self):
        with pytest.raises(ValueError, match="cache_salt"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="m",
                kv_rank=0,
                cache_salt="alice@bob",
            )

    def test_reject_leading_at_in_salt(self):
        with pytest.raises(ValueError, match="cache_salt"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="m",
                kv_rank=0,
                cache_salt="@user",
            )

    def test_reject_slash_in_salt(self):
        with pytest.raises(ValueError, match="cache_salt"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="m",
                kv_rank=0,
                cache_salt="tenant/alice",
            )

    def test_reject_backslash_in_salt(self):
        with pytest.raises(ValueError, match="cache_salt"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="m",
                kv_rank=0,
                cache_salt="tenant\\alice",
            )

    def test_reject_nul_in_salt(self):
        with pytest.raises(ValueError, match="cache_salt"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="m",
                kv_rank=0,
                cache_salt="bad\x00salt",
            )

    def test_reject_too_long_salt(self):
        with pytest.raises(ValueError, match="max length"):
            ObjectKey(
                chunk_hash=b"\x00",
                model_name="m",
                kv_rank=0,
                cache_salt="x" * 129,
            )

    def test_max_length_salt_accepted(self):
        key = ObjectKey(
            chunk_hash=b"\x00",
            model_name="m",
            kv_rank=0,
            cache_salt="x" * 128,
        )
        assert len(key.cache_salt) == 128

    def test_empty_salt_is_accepted(self):
        # Default (unsalted) path.
        key = ObjectKey(chunk_hash=b"\x00", model_name="m", kv_rank=0)
        assert key.cache_salt == ""

    def test_non_salt_chars_are_accepted(self):
        # Common identifier chars are fine.
        key = ObjectKey(
            chunk_hash=b"\x00",
            model_name="m",
            kv_rank=0,
            cache_salt="user-abc_123.xyz:42",
        )
        assert key.cache_salt == "user-abc_123.xyz:42"


class TestObjectKeyIsolation:
    """cache_salt must participate in eq/hash so the L1/L2 caches treat
    same-content/different-user entries as distinct."""

    def test_different_salts_are_unequal(self):
        base = {"chunk_hash": b"x", "model_name": "m", "kv_rank": 0}
        a = ObjectKey(**base, cache_salt="alice")
        b = ObjectKey(**base, cache_salt="bob")
        assert a != b
        assert hash(a) != hash(b)

    def test_empty_salt_is_unequal_to_any_salted(self):
        base = {"chunk_hash": b"x", "model_name": "m", "kv_rank": 0}
        unsalted = ObjectKey(**base)
        salted = ObjectKey(**base, cache_salt="alice")
        assert unsalted != salted

    def test_same_salt_are_equal(self):
        base = {"chunk_hash": b"x", "model_name": "m", "kv_rank": 0}
        a = ObjectKey(**base, cache_salt="alice")
        b = ObjectKey(**base, cache_salt="alice")
        assert a == b
        assert hash(a) == hash(b)


# =============================================================================
# Event Fd Interface Tests
# =============================================================================


class TestEventFdInterface:
    def test_three_distinct_event_fds(self, adapter):
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        assert store_fd >= 0
        assert lookup_fd >= 0
        assert load_fd >= 0
        assert len({store_fd, lookup_fd, load_fd}) == 3


# =============================================================================
# Store Interface Tests
# =============================================================================


class TestStoreInterface:
    def test_submit_store_returns_task_id(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        task_id = adapter.submit_store_task([key], [obj])
        assert isinstance(task_id, int)

    def test_store_signals_event_fd_and_completes(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adapter.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id].is_successful()

    def test_pop_clears_completed_tasks(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()

        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)

        completed1 = adapter.pop_completed_store_tasks()
        assert len(completed1) == 1

        completed2 = adapter.pop_completed_store_tasks()
        assert len(completed2) == 0

    def test_multiple_store_tasks_get_unique_ids(self, adapter):
        store_fd = adapter.get_store_event_fd()
        task_ids = set()
        for i in range(5):
            key = create_object_key(i)
            obj = create_memory_obj(fill_value=float(i))
            task_ids.add(adapter.submit_store_task([key], [obj]))

        assert len(task_ids) == 5

        # Wait for all completions
        completed = {}
        while len(completed) < 5:
            wait_for_event_fd(store_fd, timeout=5.0)
            completed.update(adapter.pop_completed_store_tasks())

        for tid in task_ids:
            assert completed[tid].is_successful()

    def test_batch_store(self, adapter):
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(3)]
        store_fd = adapter.get_store_event_fd()

        task_id = adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)

        completed = adapter.pop_completed_store_tasks()
        assert completed[task_id].is_successful()


# =============================================================================
# Lookup and Lock Interface Tests
# =============================================================================


class TestLookupAndLockInterface:
    def test_lookup_nonexistent_key(self, adapter):
        key = create_object_key(999)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is False

    def test_lookup_existing_key(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store first
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Lookup
        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True

    def test_lookup_mixed_keys(self, adapter):
        existing = create_object_key(1)
        missing = create_object_key(999)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_store_task([existing], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        task_id = adapter.submit_lookup_and_lock_task(
            [existing, missing], _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(lookup_fd, timeout=5.0)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        assert bitmap.test(1) is False

    def test_query_is_one_shot(self, adapter):
        key = create_object_key(1)
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)

        result1 = adapter.query_lookup_and_lock_result(task_id)
        assert result1 is not None

        result2 = adapter.query_lookup_and_lock_result(task_id)
        assert result2 is None

    def test_query_unknown_task_returns_none(self, adapter):
        assert adapter.query_lookup_and_lock_result(99999) is None


# =============================================================================
# Unlock Interface Tests
# =============================================================================


class TestUnlockInterface:
    def test_unlock_does_not_raise(self, adapter):
        key = create_object_key(1)
        adapter.submit_unlock([key])  # should not raise

    def test_unlock_after_lock(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)
        adapter.query_lookup_and_lock_result(task_id)

        adapter.submit_unlock([key])  # should not raise


# =============================================================================
# Load Interface Tests
# =============================================================================


class TestLoadInterface:
    def test_submit_load_returns_task_id(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        task_id = adapter.submit_load_task([key], [obj])
        assert isinstance(task_id, int)

    def test_load_signals_event_fd(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        adapter.submit_load_task([key], [obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)

    def test_load_existing_key_copies_data(self, adapter):
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
        assert wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True

        # Verify data was copied into the load buffer
        assert torch.all(load_obj.tensor == 42.0)

    def test_load_nonexistent_key_fails(self, adapter):
        key = create_object_key(999)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        task_id = adapter.submit_load_task([key], [obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)

        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        # Batch GET failed → no bits set
        assert bitmap.test(0) is False

    def test_query_load_is_one_shot(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        load_fd = adapter.get_load_event_fd()

        task_id = adapter.submit_load_task([key], [obj])
        wait_for_event_fd(load_fd, timeout=5.0)

        result1 = adapter.query_load_result(task_id)
        assert result1 is not None

        result2 = adapter.query_load_result(task_id)
        assert result2 is None

    def test_query_unknown_task_returns_none(self, adapter):
        assert adapter.query_load_result(99999) is None


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflow:
    def test_store_lookup_load_workflow(self, adapter):
        key = create_object_key(1)
        store_obj = create_memory_obj(size=256, fill_value=123.0)
        load_obj = create_memory_obj(size=256, fill_value=0.0)

        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store
        store_tid = adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd, timeout=5.0)
        assert adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Lookup
        lookup_tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is True

        # Load
        load_tid = adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd, timeout=5.0)
        bitmap = adapter.query_load_result(load_tid)
        assert bitmap.test(0) is True
        assert torch.all(load_obj.tensor == 123.0)

        # Unlock
        adapter.submit_unlock([key])

    def test_multiple_objects_workflow(self, adapter):
        n = 5
        keys = [create_object_key(i) for i in range(n)]
        store_objs = [
            create_memory_obj(size=64, fill_value=float(i * 10)) for i in range(n)
        ]
        load_objs = [create_memory_obj(size=64, fill_value=0.0) for _ in range(n)]

        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Store all
        store_tid = adapter.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd, timeout=5.0)
        assert adapter.pop_completed_store_tasks()[store_tid].is_successful()

        # Lookup all
        lookup_tid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(n):
            assert bitmap.test(i) is True

        # Load all
        load_tid = adapter.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd, timeout=5.0)
        bitmap = adapter.query_load_result(load_tid)
        for i in range(n):
            assert bitmap.test(i) is True
            assert torch.all(load_objs[i].tensor == float(i * 10))


# =============================================================================
# Close Tests
# =============================================================================


class TestClose:
    def test_close_does_not_raise(self):
        mock_client = MockNativeConnector()
        adp = NativeConnectorL2Adapter(mock_client)
        adp.close()

    def test_close_after_operations(self):
        mock_client = MockNativeConnector()
        adp = NativeConnectorL2Adapter(mock_client)

        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adp.get_store_event_fd()

        adp.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        adp.close()


# =============================================================================
# Config Tests
# =============================================================================


class TestRESPL2AdapterConfig:
    def test_from_dict_minimal(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        config = RESPL2AdapterConfig.from_dict(
            {
                "type": "resp",
                "host": "localhost",
                "port": 6379,
            }
        )
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.num_workers == 8
        assert config.username == ""
        assert config.password == ""

    def test_from_dict_full(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        config = RESPL2AdapterConfig.from_dict(
            {
                "type": "resp",
                "host": "10.0.0.1",
                "port": 6380,
                "num_workers": 16,
                "username": "user",
                "password": "pass",
            }
        )
        assert config.host == "10.0.0.1"
        assert config.port == 6380
        assert config.num_workers == 16
        assert config.username == "user"
        assert config.password == "pass"

    def test_from_dict_missing_host_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="host"):
            RESPL2AdapterConfig.from_dict({"type": "resp", "port": 6379})

    def test_from_dict_missing_port_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="port"):
            RESPL2AdapterConfig.from_dict({"type": "resp", "host": "localhost"})

    def test_registered_as_resp(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_registered_l2_adapter_types,
        )

        assert "resp" in get_registered_l2_adapter_types()


# =============================================================================
# NativePluginL2AdapterConfig Tests
# =============================================================================


class TestNativePluginL2AdapterConfig:
    def test_from_dict_minimal(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        config = NativePluginL2AdapterConfig.from_dict(
            {
                "type": "native_plugin",
                "module_path": "my_ext.connector",
                "class_name": "MyClient",
            }
        )
        assert config.module_path == "my_ext.connector"
        assert config.class_name == "MyClient"
        assert config.adapter_params == {}

    def test_from_dict_full(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        config = NativePluginL2AdapterConfig.from_dict(
            {
                "type": "native_plugin",
                "module_path": "my_ext.connector",
                "class_name": "MyClient",
                "adapter_params": {
                    "host": "localhost",
                    "port": 1234,
                },
            }
        )
        assert config.module_path == "my_ext.connector"
        assert config.class_name == "MyClient"
        assert config.adapter_params == {
            "host": "localhost",
            "port": 1234,
        }

    def test_from_dict_missing_module_path_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="module_path"):
            NativePluginL2AdapterConfig.from_dict(
                {
                    "type": "native_plugin",
                    "class_name": "X",
                }
            )

    def test_from_dict_missing_class_name_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="class_name"):
            NativePluginL2AdapterConfig.from_dict(
                {
                    "type": "native_plugin",
                    "module_path": "my_ext",
                }
            )

    def test_from_dict_invalid_adapter_params_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="adapter_params"):
            NativePluginL2AdapterConfig.from_dict(
                {
                    "type": "native_plugin",
                    "module_path": "my_ext",
                    "class_name": "X",
                    "adapter_params": "not_a_dict",
                }
            )

    def test_registered_as_native_plugin(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_registered_l2_adapter_types,
        )

        assert "native_plugin" in get_registered_l2_adapter_types()

    def test_help_returns_string(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import (
            NativePluginL2AdapterConfig,
        )

        h = NativePluginL2AdapterConfig.help()
        assert isinstance(h, str)
        assert "module_path" in h
        assert "class_name" in h
        assert "adapter_params" in h


# =============================================================================
# FSNativeL2AdapterConfig Tests
# =============================================================================


class TestFSNativeL2AdapterConfig:
    def test_from_dict_minimal(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        config = FSNativeL2AdapterConfig.from_dict(
            {
                "type": "fs_native",
                "base_path": "/tmp/lmcache_test",
            }
        )
        assert config.base_path == "/tmp/lmcache_test"
        assert config.num_workers == 4
        assert config.relative_tmp_dir == ""
        assert config.use_odirect is False
        assert config.read_ahead_size is None

    def test_from_dict_full(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        config = FSNativeL2AdapterConfig.from_dict(
            {
                "type": "fs_native",
                "base_path": "/data/kv_cache",
                "num_workers": 16,
                "relative_tmp_dir": ".tmp",
                "use_odirect": True,
                "read_ahead_size": 4096,
            }
        )
        assert config.base_path == "/data/kv_cache"
        assert config.num_workers == 16
        assert config.relative_tmp_dir == ".tmp"
        assert config.use_odirect is True
        assert config.read_ahead_size == 4096

    def test_from_dict_missing_base_path_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="base_path"):
            FSNativeL2AdapterConfig.from_dict({"type": "fs_native"})

    def test_from_dict_empty_base_path_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="base_path"):
            FSNativeL2AdapterConfig.from_dict({"type": "fs_native", "base_path": ""})

    def test_from_dict_invalid_num_workers_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="num_workers"):
            FSNativeL2AdapterConfig.from_dict(
                {
                    "type": "fs_native",
                    "base_path": "/tmp/x",
                    "num_workers": 0,
                }
            )

    def test_from_dict_zero_num_workers_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="num_workers"):
            FSNativeL2AdapterConfig.from_dict(
                {
                    "type": "fs_native",
                    "base_path": "/tmp/x",
                    "num_workers": -1,
                }
            )

    def test_from_dict_invalid_relative_tmp_dir_raises(
        self,
    ):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="relative_tmp_dir"):
            FSNativeL2AdapterConfig.from_dict(
                {
                    "type": "fs_native",
                    "base_path": "/tmp/x",
                    "relative_tmp_dir": 123,
                }
            )

    def test_from_dict_invalid_use_odirect_raises(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="use_odirect"):
            FSNativeL2AdapterConfig.from_dict(
                {
                    "type": "fs_native",
                    "base_path": "/tmp/x",
                    "use_odirect": "yes",
                }
            )

    def test_from_dict_invalid_read_ahead_size_raises(
        self,
    ):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="read_ahead_size"):
            FSNativeL2AdapterConfig.from_dict(
                {
                    "type": "fs_native",
                    "base_path": "/tmp/x",
                    "read_ahead_size": -1,
                }
            )

    def test_from_dict_zero_read_ahead_size_raises(
        self,
    ):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        with pytest.raises(ValueError, match="read_ahead_size"):
            FSNativeL2AdapterConfig.from_dict(
                {
                    "type": "fs_native",
                    "base_path": "/tmp/x",
                    "read_ahead_size": 0,
                }
            )

    def test_registered_as_fs_native(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_registered_l2_adapter_types,
        )

        assert "fs_native" in get_registered_l2_adapter_types()

    def test_help_returns_string(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        h = FSNativeL2AdapterConfig.help()
        assert isinstance(h, str)
        assert "base_path" in h
        assert "num_workers" in h
        assert "use_odirect" in h
        assert "read_ahead_size" in h

    def test_type_name_lookup(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_type_name_for_config,
        )
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        cfg = FSNativeL2AdapterConfig(
            base_path="/tmp/test",
        )
        assert get_type_name_for_config(cfg) == "fs_native"


# =============================================================================
# Delete Interface Tests
# =============================================================================


class TestDeleteInterface:
    def test_delete_existing_key(self, adapter):
        key = create_object_key(1)
        obj = create_memory_obj()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store
        adapter.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Verify exists
        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap.test(0) is True
        adapter.submit_unlock([key])

        # Delete (synchronous)
        adapter.delete([key])

        # Verify gone
        task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap.test(0) is False

    def test_delete_nonexistent_key(self, adapter):
        key = create_object_key(999)
        adapter.delete([key])  # should not raise

    def test_delete_empty_keys(self, adapter):
        adapter.delete([])  # should not raise

    def test_delete_batch(self, adapter):
        keys = [create_object_key(i) for i in range(5)]
        objs = [create_memory_obj(fill_value=float(i)) for i in range(5)]
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()

        # Store all
        adapter.submit_store_task(keys, objs)
        wait_for_event_fd(store_fd, timeout=5.0)
        adapter.pop_completed_store_tasks()

        # Delete first 3
        adapter.delete(keys[:3])

        # Verify: first 3 gone, last 2 remain
        task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        wait_for_event_fd(lookup_fd, timeout=5.0)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        for i in range(3):
            assert bitmap.test(i) is False
        for i in range(3, 5):
            assert bitmap.test(i) is True
        adapter.submit_unlock(keys[3:])


# =============================================================================
# Delete Backward Compatibility Tests
# =============================================================================


class TestDeleteBackwardCompatibility:
    def test_delete_noop_without_submit_batch_delete(self):
        """Connector without submit_batch_delete => delete is no-op."""

        class NoDeleteConnector:
            """Mock connector that only has the 6 original methods."""

            def __init__(self):
                self._efd = create_event_notifier()
                self._closed = False

            def event_fd(self) -> int:
                return self._efd.fileno()

            def submit_batch_get(self, keys, memoryviews):
                return 0

            def submit_batch_set(self, keys, memoryviews):
                return 0

            def submit_batch_exists(self, keys):
                return 0

            def drain_completions(self):
                return []

            def close(self):
                if not self._closed:
                    self._closed = True
                    self._efd.close()

        client = NoDeleteConnector()
        adp = NativeConnectorL2Adapter(client)
        try:
            key = create_object_key(1)
            adp.delete([key])  # should not raise, just no-op
        finally:
            adp.close()


# =============================================================================
# Usage Tracking Tests
# =============================================================================


@pytest.fixture
def adapter_with_capacity():
    """Adapter with max_capacity_gb set for usage tracking tests."""
    mock_client = MockNativeConnector()
    # 100 floats * 4 bytes = 400 bytes per obj; capacity = 2000 bytes = 2000/1024^3 GB
    adp = NativeConnectorL2Adapter(mock_client, max_capacity_gb=2000 / (1024**3))
    yield adp
    adp.close()


class TestUsageTracking:
    def test_get_usage_without_capacity(self, adapter):
        """Without max_capacity_bytes, usage_fraction == -1 (sentinel)."""
        usage = adapter.get_usage()
        assert usage.usage_fraction == -1.0
        assert usage.total_capacity_bytes == 0

    def test_get_usage_starts_at_zero(self, adapter_with_capacity):
        usage = adapter_with_capacity.get_usage()
        assert usage.usage_fraction == 0.0
        assert usage.total_bytes_used == 0

    def test_get_usage_after_store(self, adapter_with_capacity):
        adp = adapter_with_capacity
        store_fd = adp.get_store_event_fd()

        key = create_object_key(1)
        obj = create_memory_obj(size=100, fill_value=1.0)  # 100 floats = 400 bytes

        adp.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        usage = adp.get_usage()
        # 400 bytes / 2000 bytes = 0.2
        assert usage.usage_fraction == pytest.approx(0.2)
        assert usage.total_bytes_used == 400

    def test_get_usage_after_delete(self, adapter_with_capacity):
        adp = adapter_with_capacity
        store_fd = adp.get_store_event_fd()

        key = create_object_key(1)
        obj = create_memory_obj(size=100, fill_value=1.0)

        # Store
        adp.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        assert adp.get_usage().usage_fraction == pytest.approx(0.2)

        # Delete
        adp.delete([key])

        assert adp.get_usage().usage_fraction == pytest.approx(0.0)
        assert adp.get_usage().total_bytes_used == 0

    def test_get_usage_store_delete_cycle(self, adapter_with_capacity):
        adp = adapter_with_capacity
        store_fd = adp.get_store_event_fd()

        # Store 3 objects (3 * 400 = 1200 bytes)
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(size=100, fill_value=float(i)) for i in range(3)]

        adp.submit_store_task(keys, objs)
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        usage = adp.get_usage()
        assert usage.usage_fraction == pytest.approx(1200 / 2000)
        assert usage.total_bytes_used == 1200

        # Delete 2
        adp.delete(keys[:2])

        usage = adp.get_usage()
        assert usage.usage_fraction == pytest.approx(400 / 2000)
        assert usage.total_bytes_used == 400

    def test_idempotent_store_no_double_count(self, adapter_with_capacity):
        adp = adapter_with_capacity
        store_fd = adp.get_store_event_fd()

        key = create_object_key(1)
        obj = create_memory_obj(size=100, fill_value=1.0)

        # Store same key twice
        adp.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        adp.submit_store_task([key], [obj])
        wait_for_event_fd(store_fd, timeout=5.0)
        adp.pop_completed_store_tasks()

        # Should only count once
        usage = adp.get_usage()
        assert usage.usage_fraction == pytest.approx(0.2)
        assert usage.total_bytes_used == 400
