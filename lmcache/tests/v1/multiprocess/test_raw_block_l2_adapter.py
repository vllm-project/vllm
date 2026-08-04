# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    install_native_storage_ops_fallback,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_file,
    memory_obj_bytes,
    wait_for_event_fd,
)

install_native_storage_ops_fallback()
pytest.importorskip("lmcache_rust_raw_block_io")

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc  # noqa: E402
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (  # noqa: E402
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _make_adapter(tmp_path: Path) -> RawBlockL2Adapter:
    path = make_raw_block_file(tmp_path)
    config = RawBlockL2AdapterConfig(
        device_path=str(path),
        capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_enable_periodic=False,
        meta_idle_quiet_ms=0,
        io_engine="posix",
        iouring_queue_depth=8,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )
    return RawBlockL2Adapter(config)


def test_raw_block_l2_adapter_store_lookup_load_roundtrip(tmp_path):
    adapter = _make_adapter(tmp_path)
    try:
        key = make_object_key(1)
        missing_key = make_object_key(999)
        payload = b"raw-block-l2-adapter-payload"

        store_task_id = adapter.submit_store_task([key], [make_memory_obj(payload)])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        store_result = adapter.pop_completed_store_tasks()[store_task_id]
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == RAW_BLOCK_CI_SLOT_BYTES

        lookup_task_id = adapter.submit_lookup_and_lock_task(
            [key, missing_key], _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.test(0) is True
        assert lookup_bitmap.test(1) is False

        loaded = make_empty_memory_obj(len(payload))
        missing = make_empty_memory_obj(len(payload))
        load_task_id = adapter.submit_load_task([key, missing_key], [loaded, missing])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        load_bitmap = adapter.query_load_result(load_task_id)
        assert load_bitmap is not None
        assert load_bitmap.test(0) is True
        assert load_bitmap.test(1) is False
        assert memory_obj_bytes(loaded) == payload

        adapter.submit_unlock([key])
    finally:
        adapter.close()


def test_raw_block_l2_adapter_delete_makes_key_miss(tmp_path):
    adapter = _make_adapter(tmp_path)
    try:
        key = make_object_key(2)
        payload = b"delete-from-raw-block-l2"

        store_task_id = adapter.submit_store_task([key], [make_memory_obj(payload)])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        store_result = adapter.pop_completed_store_tasks()[store_task_id]
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == RAW_BLOCK_CI_SLOT_BYTES

        adapter.delete([key])

        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.test(0) is False
    finally:
        adapter.close()
