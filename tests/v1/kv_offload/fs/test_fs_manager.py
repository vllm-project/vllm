# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FSOffloadingManager's index, in-flight store tracking and disk
lookups."""

import os
import time
from collections.abc import Iterator

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    StoreTrigger,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.manager import FSOffloadingManager

_CTX = ReqContext(req_id="r1")
_STEP_END = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())
_TRIGGER = StoreTrigger.ON_COMPUTE


def _key(i: int) -> OffloadKey:
    return make_offload_key(f"block-{i}".encode(), 0)


def _file_mapper(root_dir: str) -> FileMapper:
    return FileMapper(
        root_dir=root_dir,
        model_name="test",
        tokens_per_hash=16,
        blocks_per_file=1,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype="float16",
    )


@pytest.fixture
def mapper(tmp_path) -> FileMapper:
    return _file_mapper(str(tmp_path))


@pytest.fixture
def manager(mapper) -> Iterator[FSOffloadingManager]:
    mgr = FSOffloadingManager(mapper, index_size=2)
    yield mgr
    mgr.shutdown()


def _write_file(mapper: FileMapper, key: OffloadKey) -> None:
    path = mapper.get_file_name(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"kv")


def _store(manager: FSOffloadingManager, key: OffloadKey, success=True) -> None:
    manager.prepare_store([key], _CTX, trigger=_TRIGGER)
    manager.complete_store([key], _CTX, trigger=_TRIGGER, success=success)


def _resolve(
    manager: FSOffloadingManager, key: OffloadKey, timeout: float = 10.0
) -> LookupResult:
    """Drive scheduler steps until the async disk probe for key resolves."""
    deadline = time.monotonic() + timeout
    while (result := manager.lookup(key, _CTX)) is LookupResult.RETRY:
        assert time.monotonic() < deadline, "disk lookup did not resolve"
        manager.on_schedule_end(_STEP_END)
        time.sleep(0.01)
    return result


def test_lookup_probes_disk(manager, mapper):
    """Unknown keys are probed on disk asynchronously: RETRY until the probe
    lands, then HIT for files another instance wrote, MISS otherwise."""
    on_disk, absent = _key(1), _key(2)
    _write_file(mapper, on_disk)

    assert manager.lookup(on_disk, _CTX) is LookupResult.RETRY
    assert _resolve(manager, on_disk) is LookupResult.HIT
    assert _resolve(manager, absent) is LookupResult.MISS


def test_disk_hit_is_remembered(manager, mapper):
    """A key found on disk is indexed, so later lookups hit without probing,
    even after its file is removed."""
    key = _key(1)
    _write_file(mapper, key)
    assert _resolve(manager, key) is LookupResult.HIT

    os.remove(mapper.get_file_name(key))
    manager.on_request_finished(_CTX)
    assert manager.lookup(key, _CTX) is LookupResult.HIT


def test_in_flight_store_is_pending_and_not_restored(manager):
    """A key whose store is in flight looks up as HIT_PENDING and is not
    handed out to a second store job."""
    key = _key(1)
    first = manager.prepare_store([key], _CTX, trigger=_TRIGGER)
    assert first is not None and first.keys_to_store == [key]

    assert manager.lookup(key, _CTX) is LookupResult.HIT_PENDING
    second = manager.prepare_store([key], _CTX, trigger=_TRIGGER)
    assert second is not None and second.keys_to_store == []


def test_successful_store_hits_and_is_not_restored(manager):
    """A completed store is a HIT without a disk probe (no file exists here)
    and is skipped by later stores."""
    key = _key(1)
    _store(manager, key)

    assert manager.lookup(key, _CTX) is LookupResult.HIT
    output = manager.prepare_store([key], _CTX, trigger=_TRIGGER)
    assert output is not None and output.keys_to_store == []


def test_failed_store_is_forgotten(manager):
    """A failed store is neither pending nor indexed: lookups go back to disk
    and the key can be stored again."""
    key = _key(1)
    _store(manager, key, success=False)

    assert _resolve(manager, key) is LookupResult.MISS
    output = manager.prepare_store([key], _CTX, trigger=_TRIGGER)
    assert output is not None and output.keys_to_store == [key]


def test_index_evicts_least_recently_used(manager):
    """The index is a bounded LRU: a lookup refreshes a key, and the oldest
    key is dropped once index_size is exceeded."""
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _store(manager, k1)
    _store(manager, k2)
    assert manager.lookup(k1, _CTX) is LookupResult.HIT  # k2 is now oldest

    _store(manager, k3)

    assert manager.lookup(k1, _CTX) is LookupResult.HIT
    assert manager.lookup(k3, _CTX) is LookupResult.HIT
    assert _resolve(manager, k2) is LookupResult.MISS


def test_reset_cache_drops_index_and_in_flight_stores(manager):
    """After reset_cache, neither indexed nor in-flight keys are remembered,
    so both can be stored again."""
    stored, in_flight = _key(1), _key(2)
    _store(manager, stored)
    manager.prepare_store([in_flight], _CTX, trigger=_TRIGGER)

    manager.reset_cache()

    output = manager.prepare_store([stored, in_flight], _CTX, trigger=_TRIGGER)
    assert output is not None
    assert output.keys_to_store == [stored, in_flight]
