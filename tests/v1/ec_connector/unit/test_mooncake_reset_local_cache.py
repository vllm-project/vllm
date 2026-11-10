# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reset of ECMooncake local cache after reset_encoder_cache."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeLoadSpec,
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
    def __init__(self, request_id: str, mm_hash: str):
        self.request_id = request_id
        self.mm_features = [_Feat(mm_hash, _Pos())]
        self.ec_transfer_params: dict = {}

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
    conn._transfers = SchedulerTransferTable(8, 300)
    conn._drain_pending = False
    conn._drain_push_notifications = lambda: None  # type: ignore[method-assign]
    conn._note_awaiting_push = lambda *a, **k: None  # type: ignore[method-assign]
    return conn


def _load_spec(mm_hash: str, transfer_id: str = "xfer-a") -> ECMooncakeLoadSpec:
    return ECMooncakeLoadSpec(
        mm_hash=mm_hash,
        nbytes=64,
        shape=(8, 4),
        dtype="float16",
        transfer_id=transfer_id,
    )


def test_scheduler_reset_encoder_cache_resets_connector():
    sched = Scheduler.__new__(Scheduler)
    sched.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    sched.encoder_cache_manager.cached["hashA"] = {"r1"}
    connector = MagicMock()
    sched.ec_connector = connector

    Scheduler.reset_encoder_cache(sched)

    assert sched.encoder_cache_manager.cached == {}
    connector.reset.assert_called_once_with()


def test_reset_clears_stale_local_cache_and_plans_load():
    conn = _consumer_scheduler()
    req = _Req("r1", "hashA")
    conn.update_state_after_alloc(req, 0)
    assert "hashA" in conn._local_cache

    # A completed Load leaves the transfer READY, which would also skip Load.
    conn._transfers.observe_ready(_load_spec("hashA"), deadline=1e18)
    record = conn._transfers.begin_load("hashA", "xfer-a", "r1", deadline=1e18)
    assert record is not None
    conn._transfers.take_loads_to_dispatch()
    assert conn._transfers.complete_load("hashA")
    assert conn._transfers.has_state("hashA", (SchedulerTransferState.READY,))

    ecm = EncoderCacheManager(cache_size=16)
    ecm.cached["hashA"] = {"r1"}
    ecm.reset()
    conn.reset()

    assert "hashA" not in conn._local_cache
    assert "hashA" not in ecm.cached
    assert conn._transfers.has_state("hashA", (SchedulerTransferState.AVAILABLE,))
    assert not conn._transfers.has_state("hashA", (SchedulerTransferState.READY,))

    ready = conn.ensure_cache_available(_Req("r2", "hashA"), 0)
    assert ready is False
    assert conn._scheduler_pending_work is True
    assert conn._transfers.has_state("hashA", (SchedulerTransferState.LOADING,))
    dispatched = conn._transfers.take_loads_to_dispatch()
    assert [rec.mm_hash for rec in dispatched] == ["hashA"]


def test_reset_without_transfer_record_does_not_false_hit():
    conn = _consumer_scheduler()
    conn.update_state_after_alloc(_Req("r1", "hashA"), 0)
    conn.reset()

    ready = conn.ensure_cache_available(_Req("r2", "hashA"), 0)
    assert ready is False
    assert "hashA" not in conn._local_cache
    assert not conn._transfers.has_state("hashA", (SchedulerTransferState.LOADING,))
