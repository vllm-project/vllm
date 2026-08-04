# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import patch
import os
import select
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


requires_raw_block_ext = pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)


class _RecordingListener(L2AdapterListener):
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


class _FailingListener(L2AdapterListener):
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        del keys
        raise RuntimeError("store listener failed")

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        raise RuntimeError("access listener failed")

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        del keys
        raise RuntimeError("delete listener failed")


def _create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _create_memory_obj(size: int = 1024, fill_value: float = 0.0) -> TensorMemoryObj:
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


def _create_complex_memory_obj(
    size: int = 1024,
    fill_value: complex = 0j,
) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.complex64)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.complex64,
        address=0,
        phy_size=raw_data.numel() * raw_data.element_size(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _wait_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            os.eventfd_read(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def _make_config(
    device_path: str,
    *,
    slot_bytes: int = 64 * 1024,
    capacity_bytes: int = 0,
    io_engine: str = "posix",
    use_uring_cmd: bool = False,
) -> RawBlockL2AdapterConfig:
    return RawBlockL2AdapterConfig(
        device_path=device_path,
        slot_bytes=slot_bytes,
        capacity_bytes=capacity_bytes,
        use_odirect=False,
        io_engine=io_engine,
        use_uring_cmd=use_uring_cmd,
        block_align=4096,
        header_bytes=4096,
        meta_total_bytes=1 * 1024 * 1024,
        meta_enable_periodic=False,
        num_store_workers=2,
        num_lookup_workers=1,
        num_load_workers=2,
    )


def _config_dict(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "device_path": "/tmp/raw-block-test-device",
        "slot_bytes": 64 * 1024,
        "use_odirect": False,
    }
    config.update(overrides)
    return config


def test_raw_block_l2_adapter_config_default_io_engine():
    config = RawBlockL2AdapterConfig.from_dict(_config_dict())

    assert config.io_engine == "posix"


@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
def test_raw_block_l2_adapter_config_accepts_io_engine_values(io_engine):
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine=io_engine))

    assert config.io_engine == io_engine


def test_raw_block_l2_adapter_config_rejects_invalid_io_engine():
    with pytest.raises(ValueError, match="io_engine"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine="uring"))


@pytest.mark.parametrize("legacy_key", ["use_iouring", "use_uring"])
def test_raw_block_l2_adapter_config_legacy_use_uring_maps_to_iouring(legacy_key):
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(**{legacy_key: True}))

    assert config.io_engine == "io_uring"


def test_raw_block_l2_adapter_config_explicit_io_engine_wins_over_legacy_flag():
    config = RawBlockL2AdapterConfig.from_dict(
        _config_dict(io_engine="posix", use_iouring=True)
    )

    assert config.io_engine == "posix"


def test_raw_block_l2_adapter_config_validates_iouring_queue_depth():
    with pytest.raises(ValueError, match="iouring_queue_depth"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(iouring_queue_depth=0))


@pytest.mark.parametrize("block_align", [0, -1, 3, 4095])
def test_raw_block_l2_adapter_config_rejects_non_power_of_2_block_align(
    block_align: int,
) -> None:
    """from_dict rejects invalid block_align values."""
    with pytest.raises(ValueError, match="block_align"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(block_align=block_align))


def _run_store(adapter: RawBlockL2Adapter, keys, objects) -> bool:
    task_id = adapter.submit_store_task(keys, objects)
    assert _wait_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert task_id in completed
    return completed[task_id].is_successful()


def _run_lookup(adapter: RawBlockL2Adapter, keys):
    task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
    assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
    return task_id, adapter.query_lookup_and_lock_result(task_id)


def _run_load(adapter: RawBlockL2Adapter, keys, objects):
    task_id = adapter.submit_load_task(keys, objects)
    assert _wait_event_fd(adapter.get_load_event_fd())
    return task_id, adapter.query_load_result(task_id)


def test_raw_block_l2_adapter_config_parses_uring_flags():
    cfg = RawBlockL2AdapterConfig.from_dict(
        {
            "type": "raw_block",
            "device_path": "/tmp/raw-block-dev",
            "slot_bytes": 64 * 1024,
            "use_odirect": False,
            "io_engine": "io_uring",
        }
    )

    assert cfg.io_engine == "io_uring"
    assert cfg.use_uring_cmd is False

    with pytest.raises(ValueError, match="use_uring_cmd requires io_uring"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/tmp/raw-block-dev",
                "slot_bytes": 64 * 1024,
                "use_uring_cmd": True,
            }
        )


def test_raw_block_l2_adapter_uring_cmd_rejects_regular_file():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        with pytest.raises(ValueError, match="NVMe namespace character device"):
            RawBlockL2Adapter(
                _make_config(
                    dev_path,
                    io_engine="io_uring",
                    use_uring_cmd=True,
                )
            )


@requires_raw_block_ext
def test_raw_block_l2_adapter_store_lookup_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key1 = _create_object_key(1)
            key_miss = _create_object_key(2)
            key3 = _create_object_key(3)
            obj1 = _create_memory_obj(fill_value=1.0)
            obj3 = _create_memory_obj(fill_value=3.0)

            assert _run_store(adapter, [key1, key3], [obj1, obj3]) is True

            lookup_task_id, lookup_bitmap = _run_lookup(
                adapter,
                [key1, key_miss, key3],
            )
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_lookup_and_lock_result(lookup_task_id) is None

            load_buffers = [
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
            ]
            load_task_id, load_bitmap = _run_load(
                adapter,
                [key1, key_miss, key3],
                load_buffers,
            )
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_load_result(load_task_id) is None
            assert torch.equal(load_buffers[0].tensor, obj1.tensor)
            assert torch.equal(load_buffers[2].tensor, obj3.tensor)
            assert torch.count_nonzero(load_buffers[1].tensor) == 0

            adapter.submit_unlock([key1, key_miss, key3])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_delete_respects_lock_until_unlock():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key = _create_object_key(11)
            obj = _create_memory_obj(fill_value=11.0)
            assert _run_store(adapter, [key], [obj]) is True

            _, bitmap = _run_lookup(adapter, [key])
            assert bitmap is not None
            assert bitmap.get_indices_list() == [0]

            adapter.delete([key])
            _, still_present = _run_lookup(adapter, [key])
            assert still_present is not None
            assert still_present.get_indices_list() == [0]
            adapter.submit_unlock([key, key])

            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
            adapter.submit_unlock([key])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_uses_global_eviction_accounting():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        capacity_bytes = (1 * 1024 * 1024) + slot_bytes
        adapter = RawBlockL2Adapter(
            _make_config(
                dev_path,
                slot_bytes=slot_bytes,
                capacity_bytes=capacity_bytes,
            )
        )
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key1 = _create_object_key(21)
            key2 = _create_object_key(22)
            obj1 = _create_memory_obj(fill_value=21.0)
            obj2 = _create_memory_obj(fill_value=22.0)

            assert _run_store(adapter, [key1], [obj1]) is True
            assert _run_store(adapter, [key2], [obj2]) is False

            assert listener.stored == [[key1]]
            assert listener.deleted == []

            usage = adapter.get_usage()
            assert usage.total_bytes_used == slot_bytes
            assert usage.total_capacity_bytes == slot_bytes
            assert 0.0 < usage.usage_fraction <= 1.0
            assert adapter.supports_global_eviction is True

            status = adapter.report_status()
            assert status["is_healthy"] is True
            assert status["type"] == "RawBlockL2Adapter"
            assert status["core"]["usable_capacity_bytes"] == slot_bytes

            _, bitmap1 = _run_lookup(adapter, [key1])
            assert bitmap1 is not None
            assert bitmap1.get_indices_list() == [0]
            _, bitmap2 = _run_lookup(adapter, [key2])
            assert bitmap2 is not None
            assert bitmap2.get_indices_list() == []
            adapter.submit_unlock([key1, key2])

            adapter.delete([key1])
            assert listener.deleted[-1] == [key1]
            assert adapter.get_usage().total_bytes_used == 0

            assert _run_store(adapter, [key2], [obj2]) is True
            assert listener.stored[-1] == [key2]
            _, bitmap_after_delete = _run_lookup(adapter, [key1, key2])
            assert bitmap_after_delete is not None
            assert bitmap_after_delete.get_indices_list() == [1]
            adapter.submit_unlock([key1, key2])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_does_not_notify_duplicate_store():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key = _create_object_key(25)
            obj = _create_memory_obj(fill_value=25.0)

            assert _run_store(adapter, [key], [obj]) is True
            assert _run_store(adapter, [key], [obj]) is True

            assert listener.stored == [[key]]
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_listener_errors_do_not_block_eventfds():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        adapter.register_listener(_FailingListener())

        try:
            key = _create_object_key(29)
            obj = _create_memory_obj(fill_value=29.0)

            store_task_id = adapter.submit_store_task([key], [obj])
            assert _wait_event_fd(adapter.get_store_event_fd())
            assert adapter.pop_completed_store_tasks()[store_task_id].is_successful()

            load_buffer = _create_memory_obj(fill_value=0.0)
            load_task_id = adapter.submit_load_task([key], [load_buffer])
            assert _wait_event_fd(adapter.get_load_event_fd())
            load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovery_from_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(31)
        obj = _create_memory_obj(fill_value=31.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            _, lookup_bitmap = _run_lookup(adapter2, [key])
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0]

            load_buffer = _create_memory_obj(fill_value=0.0)
            _, load_bitmap = _run_load(adapter2, [key], [load_buffer])
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            assert torch.equal(load_buffer.tensor, obj.tensor)
            adapter2.submit_unlock([key])
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovery_seeds_usage_by_cache_salt():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        config = _make_config(dev_path, slot_bytes=slot_bytes)
        key = _create_object_key(33, cache_salt="u1")
        obj = _create_memory_obj(fill_value=33.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            usage = adapter2.get_usage()
            assert usage.total_bytes_used == slot_bytes
            assert dict(usage.bytes_by_cache_salt) == {"u1": slot_bytes}

            adapter2.delete([key])
            usage = adapter2.get_usage()
            assert usage.total_bytes_used == 0
            assert dict(usage.bytes_by_cache_salt) == {}
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovered_keys_seed_l2_eviction_state():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(34)
        obj = _create_memory_obj(fill_value=34.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            state = L2AdapterEvictionState(
                0,
                adapter2,
                EvictionConfig(eviction_policy="LRU", eviction_ratio=1.0),
            )
            assert state.eviction_policy.get_eviction_candidates(1) == [key]

            actions = state.eviction_policy.get_eviction_actions(1.0)
            assert len(actions) == 1
            assert actions[0].keys == [key]
            adapter2.delete(actions[0].keys)

            assert adapter2.get_usage().total_bytes_used == 0
            assert state.eviction_policy.get_eviction_candidates(1) == []
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovers_unknown_checkpoint_dtype():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(35)
        obj = _create_complex_memory_obj(fill_value=1 + 2j)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            load_buffer = _create_complex_memory_obj(fill_value=0j)
            _, load_bitmap = _run_load(adapter2, [key], [load_buffer])
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            assert load_buffer.metadata.dtype is torch.complex64
            assert torch.equal(load_buffer.tensor, obj.tensor)
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_error_bitmaps_keep_submitted_size():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            keys = [_create_object_key(41), _create_object_key(42)]
            objects = [_create_memory_obj(), _create_memory_obj()]

            with patch.object(
                adapter, "_run_lookup_task", side_effect=RuntimeError("lookup failed")
            ):
                lookup_task_id = adapter.submit_lookup_and_lock_task(
                    keys, _EMPTY_LAYOUT
                )
                assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
                lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
            assert lookup_bitmap is not None
            assert str(lookup_bitmap) == "00"

            with patch.object(
                adapter, "_run_load_task", side_effect=RuntimeError("load failed")
            ):
                load_task_id = adapter.submit_load_task(keys, objects)
                assert _wait_event_fd(adapter.get_load_event_fd())
                load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert str(load_bitmap) == "00"
        finally:
            adapter.close()
