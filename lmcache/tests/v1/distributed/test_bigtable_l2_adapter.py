# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for BigtableL2Adapter.
"""

# Standard
from contextvars import ContextVar
from unittest import mock
import asyncio
import threading
import time

# Third Party
from google.api_core.exceptions import DeadlineExceeded
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter import (
    BigtableL2Adapter,
    BigtableL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# Setup test constants
TEST_PROJECT = "test-project"
TEST_INSTANCE = "test-instance"
TEST_TABLE = "test-table"

# Define in-memory state for mock Bigtable
_BACKEND_DATA: dict[bytes, dict[bytes, bytes]] = {}
_BACKEND_LOCK = threading.Lock()
_BACKEND_ERROR: ContextVar[Exception | None] = ContextVar("backend_error", default=None)


class FakeCell:
    def __init__(
        self, family: str, qualifier: bytes, value: bytes, timestamp_micros: int = 0
    ):
        self.family = family
        self.qualifier = qualifier
        self.value = value
        self.timestamp_micros = timestamp_micros


class FakeRow:
    def __init__(self, key: bytes, cells: list[FakeCell]):
        self.row_key = key
        self.cells = cells


class FakeTable:
    async def mutate_row(self, row_key, mutations, *, operation_timeout=None, **kwargs):
        err = _BACKEND_ERROR.get()
        if err:
            raise err

        # Standard
        import time

        with _BACKEND_LOCK:
            if not isinstance(mutations, list):
                mutations = [mutations]
            for mut in mutations:
                mut_type = mut.__class__.__name__
                if mut_type == "SetCell":
                    if row_key not in _BACKEND_DATA:
                        _BACKEND_DATA[row_key] = {}
                    ts = (
                        mut.timestamp_micros
                        if getattr(mut, "timestamp_micros", None) is not None
                        else (time.time_ns() // 1000)
                    )
                    _BACKEND_DATA[row_key][mut.qualifier] = (mut.new_value, ts)
                elif mut_type == "DeleteAllFromRow":
                    _BACKEND_DATA.pop(row_key, None)

    async def bulk_mutate_rows(self, entries, *, operation_timeout=None, **kwargs):
        err = _BACKEND_ERROR.get()
        if err:
            raise err

        # Standard
        import time

        results = []
        for entry in entries:
            with _BACKEND_LOCK:
                for mut in entry.mutations:
                    mut_type = mut.__class__.__name__
                    if mut_type == "SetCell":
                        if entry.row_key not in _BACKEND_DATA:
                            _BACKEND_DATA[entry.row_key] = {}
                        ts = (
                            mut.timestamp_micros
                            if getattr(mut, "timestamp_micros", None) is not None
                            else (time.time_ns() // 1000)
                        )
                        _BACKEND_DATA[entry.row_key][mut.qualifier] = (
                            mut.new_value,
                            ts,
                        )
                    elif mut_type == "DeleteAllFromRow":
                        _BACKEND_DATA.pop(entry.row_key, None)
            results.append(None)
        return results

    async def read_row(
        self, row_key: bytes, *, row_filter=None, operation_timeout=None
    ):
        err = _BACKEND_ERROR.get()
        if err:
            raise err

        with _BACKEND_LOCK:
            row_data = _BACKEND_DATA.get(row_key)
        if row_data is None:
            return None

        is_strip = row_filter is not None and "Strip" in row_filter.__class__.__name__
        cells = []
        if row_data:
            for qual, val_tuple in row_data.items():
                if isinstance(val_tuple, tuple):
                    val, ts = val_tuple
                else:
                    val, ts = val_tuple, 0
                cell_val = b"" if is_strip else val
                cells.append(FakeCell("cf", qual, cell_val, ts))
        return FakeRow(row_key, cells)


class FakeBigtableDataClientAsync:
    def __init__(self, project: str, credentials=None):
        self.project = project
        self.credentials = credentials

    def get_table(self, instance_id: str, table_name: str):
        return FakeTable()

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def setup_bigtable_mocks():
    """Inject standard mock overrides into the bigtable_l2_adapter module."""
    # Reset backing store between tests
    with _BACKEND_LOCK:
        _BACKEND_DATA.clear()
    _BACKEND_ERROR.set(None)

    with (
        mock.patch(
            "lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter.BigtableDataClientAsync",
            FakeBigtableDataClientAsync,
        ),
        mock.patch(
            "lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter.google.auth.default",
            return_value=(None, "test-project"),
        ),
    ):
        yield


def create_test_config() -> BigtableL2AdapterConfig:
    return BigtableL2AdapterConfig(
        project_id=TEST_PROJECT,
        instance_id=TEST_INSTANCE,
        table_name=TEST_TABLE,
        family_name="cf",
        column_name="data",
    )


def create_test_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_id.to_bytes(4, byteorder="big"),
        model_name="test-model",
        kv_rank=0,
        object_group_id=0,
    )


def create_test_memory_obj(val: int, size: int) -> TensorMemoryObj:
    tensor = torch.full((size,), val, dtype=torch.uint8, device="cpu")
    meta = MemoryObjMetadata(
        shape=tensor.shape,
        dtype=tensor.dtype,
        address=0,
        phy_size=size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(tensor, meta, parent_allocator=None)


def wait_for_event(efd_fileno: int, timeout: float = 2.0) -> bool:
    """Helper to select on eventfd for completion notification."""
    # Standard
    import select

    r, _, _ = select.select([efd_fileno], [], [], timeout)
    if r:
        consume_fd(efd_fileno)
        return True
    return False


class TestBigtableL2Adapter:
    def test_registration_bigtable_adapter_registers_and_builds_config(self):
        config_dict = {
            "type": "bigtable",
            "bigtable_project_id": TEST_PROJECT,
            "bigtable_instance_id": TEST_INSTANCE,
            "bigtable_table_name": TEST_TABLE,
        }
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_registered_l2_adapter_types,
        )

        assert "bigtable" in get_registered_l2_adapter_types()

        # Build config from dict
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            _L2_ADAPTER_CONFIG_REGISTRY,
        )

        cfg_cls = _L2_ADAPTER_CONFIG_REGISTRY["bigtable"]
        cfg = cfg_cls.from_dict(config_dict)
        assert isinstance(cfg, BigtableL2AdapterConfig)
        assert cfg.project_id == TEST_PROJECT
        assert cfg.instance_id == TEST_INSTANCE
        assert cfg.table_name == TEST_TABLE
        assert (
            cfg.row_key_template == "{hash_prefix}@{model}@{rank}@{group}@{hash}@{salt}"
        )
        assert cfg.layer_group_size == 10
        assert cfg.num_layers == 32
        assert cfg.kv_size == 2

        # Create adapter
        adapter = create_l2_adapter(cfg)
        assert isinstance(adapter, BigtableL2Adapter)
        adapter.close()

    def test_store_and_load_successful_store_enables_correct_load(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        key1 = create_test_key(1)
        key2 = create_test_key(2)
        obj1 = create_test_memory_obj(42, 1024)
        obj2 = create_test_memory_obj(99, 2048)

        # Store tasks
        adapter.submit_store_task([key1, key2], [obj1, obj2])
        assert wait_for_event(adapter.get_store_event_fd())

        # Load tasks (with new destination buffers)
        dest1 = create_test_memory_obj(0, 1024)
        dest2 = create_test_memory_obj(0, 2048)
        load_task_id = adapter.submit_load_task([key1, key2], [dest1, dest2])
        assert wait_for_event(adapter.get_load_event_fd())

        load_results = adapter.query_load_result(load_task_id)
        assert load_results is not None
        assert load_results.test(0)
        assert load_results.test(1)

        # Confirm data values copied
        assert (dest1.tensor == 42).all()
        assert (dest2.tensor == 99).all()

        adapter.close()

    def test_lookup_and_lock_registers_correct_existence_and_increments_refcounts(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        key = create_test_key(1)
        obj = create_test_memory_obj(77, 512)

        # Store first
        adapter.submit_store_task([key], [obj])
        assert wait_for_event(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        # Lookup & Lock
        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event(adapter.get_lookup_and_lock_event_fd())
        lookup_res = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_res is not None
        assert lookup_res.test(0)

        # Check locked key status
        with adapter._lock:
            assert adapter._locked_keys[key] == 1

        # Double lock
        lookup_task_id_2 = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(lookup_task_id_2)

        with adapter._lock:
            assert adapter._locked_keys[key] == 2

        # Unlock
        adapter.submit_unlock([key])
        with adapter._lock:
            assert adapter._locked_keys[key] == 1

        adapter.submit_unlock([key])
        with adapter._lock:
            assert key not in adapter._locked_keys

        adapter.close()

    def test_delete_removes_keys_from_adapter_and_skips_locked_keys(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        key1 = create_test_key(1)
        key2 = create_test_key(2)
        obj1 = create_test_memory_obj(11, 256)
        obj2 = create_test_memory_obj(22, 256)

        # Store
        adapter.submit_store_task([key1, key2], [obj1, obj2])
        assert wait_for_event(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()

        # Lock key1 only
        lookup_task_id = adapter.submit_lookup_and_lock_task([key1], _EMPTY_LAYOUT)
        assert wait_for_event(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(lookup_task_id)

        # Delete both. key1 should be skipped because it is locked.
        class TestListener(L2AdapterListener):
            def __init__(self):
                self.deleted = []

            def on_l2_keys_stored(self, keys):
                pass

            def on_l2_keys_accessed(self, keys):
                pass

            def on_l2_keys_deleted(self, keys):
                self.deleted.extend(keys)

        listener = TestListener()
        adapter.register_listener(listener)

        adapter.delete([key1, key2])

        # Give background event loop thread a brief moment to run delete
        time.sleep(0.1)

        # key2 is deleted, key1 is not
        assert key2 in listener.deleted
        assert key1 not in listener.deleted

        # Verify Bigtable backing store contents
        with _BACKEND_LOCK:
            assert adapter._key_encoder.encode_row_key(key1) in _BACKEND_DATA
            assert adapter._key_encoder.encode_row_key(key2) not in _BACKEND_DATA

        adapter.close()

    def test_circuit_breaker_consecutive_connection_failures_disables_adapter(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        key = create_test_key(1)
        obj = create_test_memory_obj(10, 128)

        # Inject consecutive DeadlineExceeded connection errors
        _BACKEND_ERROR.set(DeadlineExceeded("Simulated gRPC Timeout"))

        # Try to store 3 times (limit is 3)
        for _ in range(3):
            adapter.submit_store_task([key], [obj])
            assert wait_for_event(adapter.get_store_event_fd())
            adapter.pop_completed_store_tasks()

        # Adapter should be disabled now
        with adapter._lock:
            assert adapter._connection_disabled

        # Subsequent store task should fail immediately without hitting Bigtable
        # Reset error token - if adapter hits backend, it would fail. But since
        # it's disabled, it short-circuits.
        _BACKEND_ERROR.set(None)
        task_id = adapter.submit_store_task([key], [obj])
        assert wait_for_event(adapter.get_store_event_fd())
        results = adapter.pop_completed_store_tasks()
        assert not results[task_id].is_successful()

        # Backing store should still be empty
        with _BACKEND_LOCK:
            assert not _BACKEND_DATA

        adapter.close()

    def test_sharder_and_key_encoder(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.bigtable_key_encoder import (
            BigtableL2KeyEncoder,
        )
        from lmcache.v1.storage_backend.connector.bigtable_sharder import (
            BigtablePayloadSharder,
        )

        # Test BigtableL2KeyEncoder
        encoder = BigtableL2KeyEncoder(
            template="{hash_prefix}@{model}@{rank}@{group}@{hash}@{salt}"
        )
        key = ObjectKey(
            chunk_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            model_name="my-model",
            kv_rank=15,
            object_group_id=3,
            cache_salt="user-salt",
        )
        row_key = encoder.encode_row_key(key)
        assert row_key == b"0102@my-model@0000000f@3@0102030405060708@user-salt"

        # Test with empty salt
        key_no_salt = ObjectKey(
            chunk_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            model_name="my-model",
            kv_rank=15,
            object_group_id=3,
        )
        row_key_no_salt = encoder.encode_row_key(key_no_salt)
        assert row_key_no_salt == b"0102@my-model@0000000f@3@0102030405060708"

        # Test with layerwise key
        # Standard
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class LayerwiseObjectKey(ObjectKey):
            layer_id: int = 0

        key_layerwise = LayerwiseObjectKey(
            chunk_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            model_name="my-model",
            kv_rank=15,
            object_group_id=3,
            cache_salt="user-salt",
            layer_id=5,
        )
        row_key_layerwise = encoder.encode_row_key(key_layerwise)
        assert (
            row_key_layerwise
            == b"0102@my-model@0000000f@3@0102030405060708@user-salt@layer_5"
        )

        # Test BigtablePayloadSharder
        sharder = BigtablePayloadSharder(num_layers=4, layer_group_size=2, kv_size=2)
        payload = b"K0K0K1K1K2K2K3K3V0V0V1V1V2V2V3V3"
        shards = sharder.shard(payload)

        assert shards["layers_0_1"] == b"K0K0K1K1V0V0V1V1"
        assert shards["layers_2_3"] == b"K2K2K3K3V2V2V3V3"

        reassembled = sharder.reassemble(shards)
        assert reassembled == payload

    def test_adapter_writes_shards_to_bigtable(self):
        cfg = create_test_config()
        cfg.num_layers = 4
        cfg.layer_group_size = 2
        cfg.kv_size = 2

        adapter = BigtableL2Adapter(cfg)
        key = create_test_key(1)
        obj = create_test_memory_obj(42, 64)

        adapter.submit_store_task([key], [obj])
        assert wait_for_event(adapter.get_store_event_fd())

        row_key = adapter._key_encoder.encode_row_key(key)
        with _BACKEND_LOCK:
            assert row_key in _BACKEND_DATA
            row_data = _BACKEND_DATA[row_key]
            assert b"layers_0_1" in row_data
            assert b"layers_2_3" in row_data
            assert len(row_data[b"layers_0_1"][0]) == 32
            assert len(row_data[b"layers_2_3"][0]) == 32

        adapter.close()

    def test_store_skips_large_writes_no_sharding(self):
        cfg = create_test_config()
        cfg.layer_group_size = 0
        cfg.max_chunk_size_mb = 0.0001  # ~105 bytes
        cfg.column_name = "data"

        adapter = BigtableL2Adapter(cfg)
        key = create_test_key(1)
        obj = create_test_memory_obj(42, 200)  # 200 bytes > 105 bytes

        adapter.submit_store_task([key], [obj])
        assert wait_for_event(adapter.get_store_event_fd())

        row_key = adapter._key_encoder.encode_row_key(key)
        with _BACKEND_LOCK:
            assert row_key not in _BACKEND_DATA

        adapter.close()

    def test_store_skips_large_writes_sharding_enabled(self):
        cfg = create_test_config()
        cfg.layer_group_size = 10

        adapter = BigtableL2Adapter(cfg)
        key = create_test_key(1)
        obj = create_test_memory_obj(42, 64)

        class FakeBytes(bytes):
            def __len__(self):
                return 95 * 1024 * 1024

        fake_shards = {"shard1": FakeBytes()}

        with mock.patch(
            "lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter._prepare_and_shard",
            return_value=(FakeBytes(), fake_shards),
        ):
            adapter.submit_store_task([key], [obj])
            assert wait_for_event(adapter.get_store_event_fd())

        row_key = adapter._key_encoder.encode_row_key(key)
        with _BACKEND_LOCK:
            assert row_key not in _BACKEND_DATA

        adapter.close()

    def test_dual_mode_write(self):
        """Verify dual-mode write in L2 adapter:
        - payload < 150MB combines mutations in a single mutate_row.
        - payload >= 150MB sends separate mutate_row calls concurrently.
        """
        cfg = create_test_config()
        cfg.num_layers = 4
        cfg.layer_group_size = 2
        cfg.kv_size = 2

        adapter = BigtableL2Adapter(cfg)
        key_small = create_test_key(1)
        obj_small = create_test_memory_obj(42, 64)

        calls = []
        original_mutate_row = FakeTable.mutate_row

        async def spy_mutate_row(self_table, row_key, mutations, **kwargs):
            calls.append((row_key, mutations))
            return await original_mutate_row(self_table, row_key, mutations, **kwargs)

        with mock.patch.object(FakeTable, "mutate_row", spy_mutate_row):
            adapter.submit_store_task([key_small], [obj_small])
            assert wait_for_event(adapter.get_store_event_fd())

        # For small write, it should be a single mutate_row call with 2 mutations
        assert len(calls) == 1
        row_key, mutations = calls[0]
        if not isinstance(mutations, list):
            mutations = [mutations]
        assert len(mutations) == 2  # 2 shards

        # 2. Test Large Write (>= 150MB)
        calls.clear()
        key_large = create_test_key(2)
        obj_large = create_test_memory_obj(42, 64)

        class FakeBytes(bytes):
            def __len__(self):
                return 160 * 1024 * 1024

        fake_bytes_obj = FakeBytes(b"fake")
        fake_shards = {
            "layers_0_1": b"fake_1",
            "layers_2_3": b"fake_2",
        }

        with (
            mock.patch(
                "lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter."
                "_prepare_and_shard",
                return_value=(fake_bytes_obj, fake_shards),
            ),
            mock.patch.object(FakeTable, "mutate_row", spy_mutate_row),
        ):
            adapter.submit_store_task([key_large], [obj_large])
            assert wait_for_event(adapter.get_store_event_fd())

        # For large write, it should make 2 separate mutate_row calls
        assert len(calls) == 2
        for rk, mutations in calls:
            if not isinstance(mutations, list):
                mutations = [mutations]
            assert len(mutations) == 1

        adapter.pop_completed_store_tasks()
        adapter.close()

    def test_circuit_breaker_resets_on_clean_cache_misses(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        # Set connection failures
        with adapter._lock:
            adapter._connection_failures = 2

        key = create_test_key(1)
        # Lookup key that doesn't exist
        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(lookup_task_id)

        # Verify failures reset to 0
        with adapter._lock:
            assert adapter._connection_failures == 0

        adapter.close()

    def test_lookup_rollback_on_cancellation(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        key1 = create_test_key(1)
        key2 = create_test_key(2)

        # Pre-set existence cache to True for key1 so it gets pre-acquired
        adapter._exists_cache.put(
            adapter._key_encoder.encode_row_key(key1).decode("utf-8"), True
        )
        # Pre-set existence cache to None for key2 so it goes to read_row
        adapter._exists_cache.invalidate(
            adapter._key_encoder.encode_row_key(key2).decode("utf-8")
        )

        async_block = asyncio.Event()

        # We mock read_row to block indefinitely
        async def mock_read_row(*args, **kwargs):
            await async_block.wait()
            return None

        def cancel_lookup():
            for task in asyncio.all_tasks(adapter._loop):
                if "_execute_lookup" in str(task):
                    task.cancel()

        with mock.patch.object(FakeTable, "read_row", mock_read_row):
            lookup_task_id = adapter.submit_lookup_and_lock_task(
                [key1, key2], _EMPTY_LAYOUT
            )
            # Allow task to run and block on read_row
            time.sleep(0.1)

            # Trigger cancellation from event loop thread
            adapter._loop.call_soon_threadsafe(cancel_lookup)

            # Wait for task completion notification on eventfd
            assert wait_for_event(adapter.get_lookup_and_lock_event_fd())
            adapter.query_lookup_and_lock_result(lookup_task_id)

        # Refcounts should be rolled back to 0
        with adapter._lock:
            assert adapter._locked_keys[key1] == 0
            assert adapter._locked_keys[key2] == 0

        adapter.close()

    def test_lookup_reconciles_accounting_on_cache_miss(self):
        cfg = create_test_config()
        adapter = BigtableL2Adapter(cfg)

        key = create_test_key(1)
        # Manually store size in adapter size accounting to simulate ghost bytes
        with adapter._lock:
            adapter._key_sizes[key] = 1024
            adapter._object_size_cache[
                adapter._key_encoder.encode_row_key(key).decode("utf-8")
            ] = 1024

        class TestListener(L2AdapterListener):
            def __init__(self):
                self.deleted = []

            def on_l2_keys_stored(self, keys):
                pass

            def on_l2_keys_accessed(self, keys):
                pass

            def on_l2_keys_deleted(self, keys):
                self.deleted.extend(keys)

        listener = TestListener()
        adapter.register_listener(listener)

        # Lookup key that doesn't exist in backend
        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event(adapter.get_lookup_and_lock_event_fd())
        adapter.query_lookup_and_lock_result(lookup_task_id)

        # Allow background threads to run callback
        time.sleep(0.1)

        # Verify key was popped and listener notified
        assert key in listener.deleted
        with adapter._lock:
            assert key not in adapter._key_sizes

        adapter.close()

    def test_unsharded_bulk_store_size_chunking(self):
        cfg = create_test_config()
        cfg.layer_group_size = 0  # unsharded
        cfg.max_chunk_size_mb = 50.0

        adapter = BigtableL2Adapter(cfg)

        key1 = create_test_key(1)
        key2 = create_test_key(2)
        key3 = create_test_key(3)

        class FakeBytes(bytes):
            def __new__(cls, *args, **kwargs):
                return bytes.__new__(cls, *args, **kwargs)

            def __len__(self):
                return 12 * 1024 * 1024

        # Submit store task with 3 objects
        obj1 = create_test_memory_obj(42, 64)
        obj2 = create_test_memory_obj(42, 64)
        obj3 = create_test_memory_obj(42, 64)

        bulk_mutate_calls = []
        original_bulk_mutate_rows = FakeTable.bulk_mutate_rows

        async def spy_bulk_mutate_rows(self_table, entries, **kwargs):
            bulk_mutate_calls.append(list(entries))
            return await original_bulk_mutate_rows(self_table, entries, **kwargs)

        with (
            mock.patch(
                "lmcache.v1.distributed.l2_adapters.bigtable_l2_adapter._prepare_bytes",
                side_effect=[FakeBytes(b"1"), FakeBytes(b"2"), FakeBytes(b"3")],
            ),
            mock.patch.object(FakeTable, "bulk_mutate_rows", spy_bulk_mutate_rows),
        ):
            adapter.submit_store_task([key1, key2, key3], [obj1, obj2, obj3])
            assert wait_for_event(adapter.get_store_event_fd())

        # With 12 MiB each, first 2 objects = 24 MiB fits in first batch.
        # Adding third object = 36 MiB > 30 MiB.
        # So first batch has 2 entries, second batch has 1 entry.
        assert len(bulk_mutate_calls) == 2
        assert len(bulk_mutate_calls[0]) == 2
        assert len(bulk_mutate_calls[1]) == 1

        adapter.pop_completed_store_tasks()
        adapter.close()
