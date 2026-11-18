# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reset of ECMooncake local cache after reset_encoder_cache."""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    MemoryAllocation,
)
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
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import ECMooncakeWorker
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu.ec_connector import ActiveECConnector, ECConnector

pytestmark = pytest.mark.cpu_test

_TRANSFER_ID = "xfer-a"
_SHAPE = (8, 4)
_DTYPE = torch.float16
_NBYTES = 64
_POOL_CAPACITY = 256
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


def _one_tensor_pool() -> ConsumerMemoryPool:
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    transfer.unregister_memory.return_value = True
    pool = ConsumerMemoryPool(_POOL_CAPACITY, transfer)
    pool.prepare(torch.device("cpu"))
    assert pool.tensor is not None
    return pool


def _reservation_manager(
    pool: ConsumerMemoryPool,
) -> ConsumerReservationManager:
    return ConsumerReservationManager(pool, lease_ttl=300, tombstone_limit=8)


def _worker_with_pool(pool: ConsumerMemoryPool) -> ECMooncakeWorker:
    worker = ECMooncakeWorker.__new__(ECMooncakeWorker)
    worker._consumer_memory = pool
    return worker


def _load_and_publish(
    manager: ConsumerReservationManager,
    transfer_id: str,
    mm_hash: str,
) -> MemoryAllocation:
    record, write = manager.reserve(
        transfer_id, mm_hash, _NBYTES, _SHAPE, "float16", _DTYPE
    )
    assert record is not None
    assert write is True
    accepted, published = manager.complete(transfer_id, record.reservation_id)
    assert accepted is True
    assert published is True
    return manager.take(transfer_id, mm_hash)


def test_invalidate_published_is_noop_before_prepare():
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    pool = ConsumerMemoryPool(_POOL_CAPACITY, transfer)
    pool.invalidate_published()
    assert pool.tensor is None


def test_reset_evicts_published_resident_and_allows_reserve_b():
    """After Load A and an idle reset, a one-tensor pool can reserve B."""
    pool = _one_tensor_pool()
    manager = _reservation_manager(pool)
    allocation_a = _load_and_publish(manager, "xfer-a", "hashA")
    assert pool.take_resident("hashA", _SHAPE, "float16") is allocation_a.tensor

    _worker_with_pool(pool).reset()

    assert pool.acquire_cached("hashA", _SHAPE, _DTYPE) is None
    assert pool.take_resident("hashA", _SHAPE, "float16") is None

    record_b, write_b = manager.reserve(
        "xfer-b", "hashB", _NBYTES, _SHAPE, "float16", _DTYPE
    )
    assert record_b is not None
    assert write_b is True
    assert record_b.state is ConsumerReservationState.WRITING
    assert record_b.allocation is not None
    assert record_b.allocation is not allocation_a


def test_reset_does_not_reuse_published_resident_as_ready():
    """After reset, reserving A under a new transfer_id must allocate again."""
    pool = _one_tensor_pool()
    manager = _reservation_manager(pool)
    allocation_a = _load_and_publish(manager, "xfer-a", "hashA")

    _worker_with_pool(pool).reset()

    record_a2, write_a2 = manager.reserve(
        "xfer-a2", "hashA", _NBYTES, _SHAPE, "float16", _DTYPE
    )
    assert record_a2 is not None
    assert write_a2 is True
    assert record_a2.state is ConsumerReservationState.WRITING
    assert record_a2.allocation is not None
    assert record_a2.allocation is not allocation_a
    assert record_a2.lease is None


def test_reset_leaves_in_flight_writing_allocation():
    pool = _one_tensor_pool()
    manager = _reservation_manager(pool)
    record, write = manager.reserve(
        "xfer-a", "hashA", _NBYTES, _SHAPE, "float16", _DTYPE
    )
    assert record is not None
    assert write is True
    allocation = record.allocation

    _worker_with_pool(pool).reset()

    assert record.state is ConsumerReservationState.WRITING
    assert record.allocation is allocation
    accepted, published = manager.complete("xfer-a", record.reservation_id)
    assert accepted is True
    assert published is True
    taken = manager.take("xfer-a", "hashA")
    assert taken is allocation


def test_connector_reset_forwards_to_worker():
    connector = ECMooncakeConnector.__new__(ECMooncakeConnector)
    connector._scheduler = None
    connector._worker = MagicMock()
    ECMooncakeConnector.reset(connector)
    connector._worker.reset.assert_called_once_with()


def test_active_ec_connector_reset_forwards_to_transfer_connector():
    wrapper = ActiveECConnector.__new__(ActiveECConnector)
    wrapper.ec_connector = MagicMock()
    ActiveECConnector.reset(wrapper)
    wrapper.ec_connector.reset.assert_called_once_with()
    ECConnector().reset()


# ---------------------------------------------------------------------------
# Deferred invalidation: leased residents survive reset only until their
# final lease is released, and never serve new work in the meantime.


_POOL_CAPACITY_2 = 512  # two 256-byte-aligned allocations


class _ToggleEvent(torch.Event):
    """Stand-in for a CUDA retire event whose completion the test controls."""

    def __init__(self) -> None:
        super().__init__()
        self.polled = False
        self.complete = False

    def query(self) -> bool:
        self.polled = True
        return self.complete


def _two_tensor_pool() -> ConsumerMemoryPool:
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    transfer.unregister_memory.return_value = True
    pool = ConsumerMemoryPool(_POOL_CAPACITY_2, transfer)
    pool.prepare(torch.device("cpu"))
    assert pool.tensor is not None
    return pool


def _reserve(
    manager: ConsumerReservationManager,
    transfer_id: str,
    mm_hash: str,
) -> tuple[ConsumerReservation | None, bool]:
    return manager.reserve(transfer_id, mm_hash, _NBYTES, _SHAPE, "float16", _DTYPE)


def _reserve_cached(
    manager: ConsumerReservationManager,
    transfer_id: str,
    mm_hash: str,
) -> ConsumerReservation:
    """Take a READY-from-cache reservation holding a lease on the resident."""
    record, write = _reserve(manager, transfer_id, mm_hash)
    assert record is not None
    assert write is False
    assert record.state is ConsumerReservationState.READY
    assert record.lease is not None
    return record


def test_reset_reclaims_leased_resident_after_cancel():
    """A leased resident skipped at reset frees up once its lease is cancelled."""
    pool = _one_tensor_pool()
    manager = _reservation_manager(pool)
    allocation_a = _load_and_publish(manager, "xfer-a", "hashA")
    leased = _reserve_cached(manager, "xfer-a2", "hashA")

    _worker_with_pool(pool).reset()

    # The lease pins the entry for now, but nothing new may reuse it and a
    # full pool still cannot serve a second destination.
    assert pool.acquire_cached("hashA", _SHAPE, _DTYPE) is None
    assert _reserve(manager, "xfer-b", "hashB")[0] is None

    assert manager.cancel("xfer-a2", leased.reservation_id) is True

    record_b, write_b = _reserve(manager, "xfer-b", "hashB")
    assert record_b is not None
    assert write_b is True
    assert record_b.allocation is not None
    assert record_b.allocation is not allocation_a


def test_reset_reclaims_leased_resident_after_expiry():
    """Expiring the outstanding lease also reclaims the invalidated resident."""
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    transfer.unregister_memory.return_value = True
    pool = ConsumerMemoryPool(_POOL_CAPACITY, transfer)
    pool.prepare(torch.device("cpu"))
    manager = ConsumerReservationManager(pool, lease_ttl=0, tombstone_limit=8)
    _load_and_publish(manager, "xfer-a", "hashA")
    _reserve_cached(manager, "xfer-a2", "hashA")

    _worker_with_pool(pool).reset()
    assert manager.expire() == 1

    record_b, write_b = _reserve(manager, "xfer-b", "hashB")
    assert record_b is not None
    assert write_b is True


def test_reset_does_not_serve_invalidated_resident_to_new_reservations():
    """After reset, reserving A again must not reuse the invalidated tensor."""
    pool = _two_tensor_pool()
    manager = _reservation_manager(pool)
    allocation_a = _load_and_publish(manager, "xfer-a", "hashA")
    _reserve_cached(manager, "xfer-a2", "hashA")

    _worker_with_pool(pool).reset()

    record_a3, write_a3 = _reserve(manager, "xfer-a3", "hashA")
    assert record_a3 is not None
    assert write_a3 is True
    assert record_a3.state is ConsumerReservationState.WRITING
    assert record_a3.lease is None
    assert record_a3.allocation is not allocation_a


def test_replacement_frees_invalidated_resident_and_keeps_new_one():
    """A fresh publish of hash A replaces the invalidated resident cleanly."""
    pool = _two_tensor_pool()
    manager = _reservation_manager(pool)
    allocation_a = _load_and_publish(manager, "xfer-a", "hashA")
    leased = _reserve_cached(manager, "xfer-a2", "hashA")

    _worker_with_pool(pool).reset()

    record_a3, write_a3 = _reserve(manager, "xfer-a3", "hashA")
    assert record_a3 is not None
    assert write_a3 is True
    manager.complete("xfer-a3", record_a3.reservation_id)
    allocation_new = manager.take("xfer-a3", "hashA")
    assert allocation_new is not allocation_a

    assert manager.cancel("xfer-a2", leased.reservation_id) is True

    lease = pool.acquire_cached("hashA", _SHAPE, _DTYPE)
    assert lease is not None
    assert lease.value is allocation_new
    pool.release_cached(lease)
    extra = pool.try_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert extra is not None
    assert extra.tensor.data_ptr() == allocation_a.tensor.data_ptr()


def test_take_rejects_invalidated_ready_lease_and_reclaims():
    """take() refuses a stale READY lease, terminates it, and frees the pool."""
    pool = _one_tensor_pool()
    manager = _reservation_manager(pool)
    _load_and_publish(manager, "xfer-a", "hashA")
    _reserve_cached(manager, "xfer-a2", "hashA")

    _worker_with_pool(pool).reset()

    with pytest.raises(RuntimeError, match="invalidated"):
        manager.take("xfer-a2", "hashA")

    assert manager.status("xfer-a2") is None
    record_b, write_b = _reserve(manager, "xfer-b", "hashB")
    assert record_b is not None
    assert write_b is True


def test_reset_evictions_are_not_reported_as_reclaimed():
    """Reset evictions skip the reclaimed report; LRU evictions keep it."""
    pool = _two_tensor_pool()
    manager = _reservation_manager(pool)
    _load_and_publish(manager, "xfer-a", "hashA")
    _load_and_publish(manager, "xfer-b", "hashB")
    leased = _reserve_cached(manager, "xfer-b2", "hashB")

    _worker_with_pool(pool).reset()

    assert pool.drain_reclaimed() == set()
    assert manager.cancel("xfer-b2", leased.reservation_id) is True
    assert pool.drain_reclaimed() == set()

    _load_and_publish(manager, "xfer-c", "hashC")
    _load_and_publish(manager, "xfer-d", "hashD")
    manager.retire_stale({}, freed=["hashC", "hashD"])
    record_e, write_e = _reserve(manager, "xfer-e", "hashE")
    assert record_e is not None
    assert write_e is True
    assert pool.drain_reclaimed() == {"hashC"}


def test_replaced_retire_event_is_not_inherited_by_new_resident():
    """Retire-event ownership across the old/new resident interleave.

    The replacement publish drops the old entry's event; the stale lease's
    final release frees the old allocation without touching the new
    resident's event; and a later LRU eviction of the new resident gates
    on that event, not the old one's.
    """
    pool = _two_tensor_pool()
    old = pool.try_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert old is not None
    pool.publish("hashA", old)
    lease = pool.acquire_cached("hashA", _SHAPE, _DTYPE)
    assert lease is not None

    stale_event = _ToggleEvent()
    pool._retire_events["hashA"] = stale_event
    pool.invalidate_published()

    fresh = pool.try_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert fresh is not None
    pool.publish("hashA", fresh)
    assert "hashA" not in pool._retire_events
    assert stale_event.polled is False

    # A CUDA retire of the new resident would record an event; stand one in.
    pool.retire_stale({}, freed=["hashA"])
    new_event = _ToggleEvent()
    pool._retire_events["hashA"] = new_event

    pool.release_cached(lease)
    assert pool._retire_events["hashA"] is new_event
    assert pool._pending_frees == []
    extra = pool.try_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert extra is not None
    assert extra.tensor.data_ptr() == old.tensor.data_ptr()

    refill = pool.reclaim_and_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert refill is None
    assert new_event.polled is True
    new_event.complete = True
    refill = pool.reclaim_and_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert refill is not None
    assert pool.drain_reclaimed() == {"hashA"}


def test_reset_drops_pending_reclaimed_notices():
    """A reclaim notice accumulated before the reset must not outlive it."""
    pool = _two_tensor_pool()
    manager = _reservation_manager(pool)
    _load_and_publish(manager, "xfer-x", "hashX")
    _load_and_publish(manager, "xfer-y", "hashY")
    manager.retire_stale({}, freed=["hashX", "hashY"])
    record_z, write_z = _reserve(manager, "xfer-z", "hashZ")
    assert record_z is not None
    assert write_z is True  # pressure evicted hashX
    assert "hashX" in pool._reclaimed  # notice pending, not yet reported

    _worker_with_pool(pool).reset()

    assert pool.drain_reclaimed() == set()

    # A post-reset LRU eviction still reports.
    _load_and_publish(manager, "xfer-p", "hashP")
    manager.retire_stale({}, freed=["hashP"])
    record_r, write_r = _reserve(manager, "xfer-r", "hashR")
    assert record_r is not None
    assert write_r is True
    assert pool.drain_reclaimed() == {"hashP"}
