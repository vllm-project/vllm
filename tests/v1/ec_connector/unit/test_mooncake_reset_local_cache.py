# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reset of ECMooncake local cache after reset_encoder_cache."""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import MemoryAllocation
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeLoadSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservation,
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    SchedulerTransferState,
    SchedulerTransferTable,
)
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

_TRANSFER_ID = "xfer-a"
_MM_HASH = "hashA"


@dataclass
class _Pos:
    offset: int = 0
    length: int = 8


@dataclass
class _Feat:
    identifier: str
    mm_position: _Pos
    modality: str = "image"


class _Req:
    def __init__(
        self,
        request_id: str,
        mm_hash: str,
        transfer_id: str | None = None,
    ):
        self.request_id = request_id
        self.mm_features = [_Feat(mm_hash, _Pos())]
        self.ec_transfer_params: dict = {}
        if transfer_id is not None:
            self.ec_transfer_params = {
                "ec_items": [{"mm_hash": mm_hash, "transfer_id": transfer_id}]
            }

    def get_num_encoder_embeds(self, index: int) -> int:
        return 8


def _consumer_scheduler() -> ECMooncakeScheduler:
    """Minimally initialize official methods without a live control plane."""
    conn = ECMooncakeScheduler.__new__(ECMooncakeScheduler)
    conn._is_producer = False
    conn._is_consumer = True
    conn._local_cache = set()
    conn._failed_saves = set()
    conn._store_candidates = []
    conn._cross_encoder_cache = False
    conn._pushes_to_prepare = {}
    conn._prepared_push_transfer_ids = set()
    conn._scheduler_pending_work = False
    conn._push_wait_timeout = 5.0
    conn._transfers = SchedulerTransferTable(1 << 20, 300)
    conn._drain_pending = False
    conn._event_ready_shards = OrderedDict()
    conn._drain_push_notifications = lambda: None  # type: ignore[method-assign]
    conn._note_awaiting_push = lambda *a, **k: None  # type: ignore[method-assign]
    return conn


def _load_spec(mm_hash: str, transfer_id: str = _TRANSFER_ID) -> ECMooncakeLoadSpec:
    return ECMooncakeLoadSpec(
        mm_hash=mm_hash,
        nbytes=64,
        shape=(8, 4),
        dtype="float16",
        transfer_id=transfer_id,
    )


def _assert_expired_and_no_load(conn: ECMooncakeScheduler, mm_hash: str) -> None:
    assert mm_hash not in conn._local_cache
    assert not conn.has_cache_item(mm_hash)
    assert conn._transfers.has_state(mm_hash, (SchedulerTransferState.EXPIRED,))
    assert not conn._transfers.has_state(
        mm_hash,
        (
            SchedulerTransferState.AVAILABLE,
            SchedulerTransferState.READY,
            SchedulerTransferState.LOADING,
            SchedulerTransferState.RESIDENT,
        ),
    )

    ready = conn.ensure_cache_available(
        _Req("r2", mm_hash, transfer_id=_TRANSFER_ID), 0
    )
    assert ready is False
    assert conn._scheduler_pending_work is False
    assert not conn._transfers.has_state(mm_hash, (SchedulerTransferState.LOADING,))
    assert conn._transfers.take_loads_to_dispatch() == []
    assert conn._transfers.begin_load(mm_hash, _TRANSFER_ID, "r2") is None


def test_scheduler_reset_encoder_cache_resets_connector():
    sched = Scheduler.__new__(Scheduler)
    sched.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    sched.encoder_cache_manager.cached["hashA"] = {"r1"}
    connector = MagicMock()
    sched.ec_connector = connector

    Scheduler.reset_encoder_cache(sched)

    assert sched.encoder_cache_manager.cached == {}
    connector.reset.assert_called_once_with()


def test_reset_expires_consumed_ready_and_does_not_plan_load():
    conn = _consumer_scheduler()
    req = _Req("r1", _MM_HASH, transfer_id=_TRANSFER_ID)
    conn.update_state_after_alloc(req, 0)
    assert _MM_HASH in conn._local_cache

    # A completed Load leaves the transfer READY. The Worker has already
    # consumed that reservation via take(); reset must expire the record
    # rather than re-arm it as a remote cache hit.
    conn._transfers.observe_ready(_load_spec(_MM_HASH), deadline=1e18)
    record = conn._transfers.begin_load(_MM_HASH, _TRANSFER_ID, "r1", deadline=1e18)
    assert record is not None
    conn._transfers.take_loads_to_dispatch()
    assert conn._transfers.complete_load(_MM_HASH)
    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.READY,))

    ecm = EncoderCacheManager(cache_size=16)
    ecm.cached[_MM_HASH] = {"r1"}
    ecm.reset()
    conn.reset()

    assert _MM_HASH not in ecm.cached
    assert not conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.READY,))
    _assert_expired_and_no_load(conn, _MM_HASH)


def test_reset_keeps_available_and_plans_load():
    """Reset clears the local cache but leaves an unconsumed reservation Load-able."""
    conn = _consumer_scheduler()
    conn.update_state_after_alloc(_Req("r1", _MM_HASH, transfer_id=_TRANSFER_ID), 0)
    conn._transfers.observe_ready(_load_spec(_MM_HASH), deadline=1e18)
    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.AVAILABLE,))
    assert conn.has_cache_item(_MM_HASH)
    assert _MM_HASH in conn._local_cache

    conn.reset()

    assert _MM_HASH not in conn._local_cache
    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.AVAILABLE,))
    assert not conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.EXPIRED,))
    assert conn.has_cache_item(_MM_HASH)

    ready = conn.ensure_cache_available(
        _Req("r2", _MM_HASH, transfer_id=_TRANSFER_ID), 0
    )
    assert ready is False
    assert conn._scheduler_pending_work is True
    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.LOADING,))
    dispatched = conn._transfers.take_loads_to_dispatch()
    assert [record.transfer_id for record in dispatched] == [_TRANSFER_ID]


def test_reset_expires_resident_and_does_not_plan_load():
    conn = _consumer_scheduler()
    conn._transfers.observe_ready(_load_spec(_MM_HASH), deadline=1e18)
    record = conn._transfers.begin_load(_MM_HASH, _TRANSFER_ID, "r1", deadline=1e18)
    assert record is not None
    conn._transfers.take_loads_to_dispatch()
    assert conn._transfers.complete_load(_MM_HASH)
    conn._transfers.release_ready(_MM_HASH, now=0.0)
    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.RESIDENT,))

    conn.reset()

    assert not conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.RESIDENT,))
    _assert_expired_and_no_load(conn, _MM_HASH)


def test_reset_leaves_loading_to_existing_async_flow():
    """Reset does not expire or re-arm an in-flight LOADING transfer.

    This does not claim in-flight reset concurrency safety; the existing
    async Load path continues to own the record.
    """
    conn = _consumer_scheduler()
    conn._transfers.observe_ready(_load_spec(_MM_HASH), deadline=1e18)
    record = conn._transfers.begin_load(_MM_HASH, _TRANSFER_ID, "r1", deadline=1e18)
    assert record is not None
    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.LOADING,))

    conn.reset()

    assert conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.LOADING,))
    assert not conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.EXPIRED,))
    assert not conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.AVAILABLE,))
    dispatched = conn._transfers.take_loads_to_dispatch()
    assert [record.transfer_id for record in dispatched] == [_TRANSFER_ID]


def test_reset_without_transfer_record_does_not_false_hit():
    conn = _consumer_scheduler()
    conn.update_state_after_alloc(_Req("r1", _MM_HASH), 0)
    conn.reset()

    ready = conn.ensure_cache_available(_Req("r2", _MM_HASH), 0)
    assert ready is False
    assert _MM_HASH not in conn._local_cache
    assert not conn.has_cache_item(_MM_HASH)
    assert not conn._transfers.has_state(_MM_HASH, (SchedulerTransferState.LOADING,))
    assert conn._transfers.take_loads_to_dispatch() == []


def test_reset_does_not_reschedule_consumed_reservation():
    """After take() consumes a reservation, reset must not plan that Load."""

    class _StubMemory:
        def __init__(self):
            self.lock = threading.RLock()

        def publish(self, mm_hash, allocation, lease=None, *, pin=True):
            return allocation

    reservations = ConsumerReservationManager(
        _StubMemory(),  # type: ignore[arg-type]
        lease_ttl=300,
        tombstone_limit=8,
    )
    allocation = MemoryAllocation(0, 64, torch.zeros((8, 4), dtype=torch.float16))
    reservations._insert(
        ConsumerReservation(
            _TRANSFER_ID,
            _MM_HASH,
            "res-a",
            ConsumerReservationState.READY,
            (8, 4),
            "float16",
            allocation,
        )
    )

    taken = reservations.take(_TRANSFER_ID, _MM_HASH)
    assert taken is allocation
    with pytest.raises(RuntimeError, match="Pushed EC tensor is not ready"):
        reservations.take(_TRANSFER_ID, _MM_HASH)

    conn = _consumer_scheduler()
    conn.update_state_after_alloc(_Req("r1", _MM_HASH, transfer_id=_TRANSFER_ID), 0)
    conn._transfers.observe_ready(_load_spec(_MM_HASH), deadline=1e18)
    record = conn._transfers.begin_load(_MM_HASH, _TRANSFER_ID, "r1", deadline=1e18)
    assert record is not None
    conn._transfers.take_loads_to_dispatch()
    assert conn._transfers.complete_load(_MM_HASH)

    conn.reset()

    _assert_expired_and_no_load(conn, _MM_HASH)
    with pytest.raises(RuntimeError, match="Pushed EC tensor is not ready"):
        reservations.take(_TRANSFER_ID, _MM_HASH)
