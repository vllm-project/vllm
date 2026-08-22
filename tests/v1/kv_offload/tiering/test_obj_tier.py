# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mock-based unit tests for ObjectStoreSecondaryTierManager.

These tests replace the NIXL backend with an in-memory mock so they run
without S3 credentials or a live object store. They verify the manager's
state machine: job submission, transfer completion polling, and lookup.
"""

import time
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig
from vllm.v1.kv_offload.tiering.obj.manager import ObjectStoreSecondaryTierManager

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


def _make_offloading_config(
    enable_kv_cache_events: bool,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> OffloadingConfig:
    if world_size is None:
        world_size = tp_size
    return OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test/model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )


_OFFLOADING_SPEC = SimpleNamespace(
    config=_make_offloading_config(enable_kv_cache_events=False),
)

_STORE_CONFIG = {
    "bucket": "mock-bucket",
    "endpoint_override": "mock:9000",
    "access_key": "mock-access",
    "secret_key": "mock-secret",
}

_BLOCK_ELEMENTS = 256
_DTYPE = torch.float32
_RUN_PREFIX = f"test/{uuid.uuid4().hex[:8]}"
_CTX = ReqContext(req_id="test-req")


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
) -> TransferJob:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=False,
        req_context=_CTX,
    )


# ---------------------------------------------------------------------------
# Mock NIXL agent
# ---------------------------------------------------------------------------


class MockNixlAgent:
    """In-memory NIXL agent. Tracks stored object keys and simulates async
    transfers: transfer() returns PROC, check_xfer_state() returns DONE and
    commits the write to the in-memory key set.

    The four methods overridden by tests (register_memory, make_prepped_xfer,
    check_xfer_state, query_memory) are stored as Callable instance attributes
    so mypy allows reassignment in tests.
    """

    # Callable attributes — tests may reassign these on instances.
    register_memory: Callable
    make_prepped_xfer: Callable
    check_xfer_state: Callable
    query_memory: Callable

    def __init__(self):
        self._stored_obj_keys: set[str] = set()
        # handle_id -> (op, [obj_keys])
        self._pending: dict[int, tuple[str, list[str]]] = {}
        self._handle_counter = 0
        self._last_obj_keys: list[str] = []
        # Bind default implementations as instance attributes.
        self.register_memory = self._register_memory
        self.make_prepped_xfer = self._make_prepped_xfer
        self.check_xfer_state = self._check_xfer_state
        self.query_memory = self._query_memory

    def create_backend(self, backend_type, params):
        pass

    def _register_memory(self, descs, mem_type=None, backends=None):
        mock = MagicMock()
        mock.trim.return_value = MagicMock()
        # Capture obj_keys from OBJ 4-tuples: (addr, len, dev_id, obj_key)
        if mem_type == "OBJ" and descs:
            self._last_obj_keys = [d[3] for d in descs if d[3]]
        return mock

    def deregister_memory(self, desc):
        pass

    def prep_xfer_dlist(self, agent_name, descs, mem_type=None, backends=None):
        return MagicMock()

    def _make_prepped_xfer(
        self,
        op,
        local_handle,
        local_indices,
        remote_handle,
        remote_indices,
        notif_msg=b"",
        backends=None,
        skip_desc_merge=False,
    ):
        handle = MagicMock()
        handle._id = self._handle_counter
        self._pending[self._handle_counter] = (op, list(self._last_obj_keys))
        self._handle_counter += 1
        return handle

    def transfer(self, handle):
        return "PROC"

    def _check_xfer_state(self, handle):
        entry = self._pending.pop(handle._id, None)
        if entry:
            op, obj_keys = entry
            if op == "WRITE":
                self._stored_obj_keys.update(obj_keys)
        return "DONE"

    def release_xfer_handle(self, handle):
        pass

    def release_dlist_handle(self, handle):
        pass

    def get_xfer_telemetry(self, handle):
        return SimpleNamespace(xferDuration=1000)

    def _query_memory(self, queries, mem_type, agent_name):
        return [object() if q[3] in self._stored_obj_keys else None for q in queries]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_events_spec(
    enable_kv_cache_events: bool,
    *,
    self_describing_kv_events: bool = False,
) -> SimpleNamespace:
    """Offloading spec stub with an explicit global KV events flag."""
    return SimpleNamespace(
        config=_make_offloading_config(enable_kv_cache_events),
        kv_events_config=OffloadingKVEventsConfig(
            enable_kv_cache_events=enable_kv_cache_events,
            self_describing_kv_events=self_describing_kv_events,
        ),
    )


def _make_tier(
    num_blocks: int = 4,
    offloading_spec: SimpleNamespace = _OFFLOADING_SPEC,
    primary_kv_view: memoryview | None = None,
    **tier_kwargs,
) -> tuple[ObjectStoreSecondaryTierManager, MockNixlAgent]:
    """Create a tier backed by a fresh MockNixlAgent."""
    mock_agent = MockNixlAgent()
    if primary_kv_view is None:
        tensor = torch.zeros((num_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
        primary_kv_view = memoryview(tensor.numpy())
    with (
        patch("vllm.v1.kv_offload.tiering.obj.manager.nixl_agent_config"),
        patch(
            "vllm.v1.kv_offload.tiering.obj.manager.nixl_agent",
            return_value=mock_agent,
        ),
    ):
        tier = ObjectStoreSecondaryTierManager(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type="obj",
            store_config=_STORE_CONFIG,
            prefix=_RUN_PREFIX,
            **tier_kwargs,
        )
    return tier, mock_agent


def drain(
    tier: ObjectStoreSecondaryTierManager, max_rounds: int = 20
) -> list[JobResult]:
    """Poll get_finished_jobs() until all in-flight jobs resolve."""
    results: list[JobResult] = []
    for _ in range(max_rounds):
        results.extend(tier.get_finished_jobs())
        if not tier._transfers:
            break
    return results


def lookup_and_wait(
    tier: ObjectStoreSecondaryTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[bool]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(locality):
    with pytest.raises(ValueError, match="Locality"):
        _make_tier(locality=locality)


class TestMockObjTierBasic:
    def setup_method(self):
        self.tier, self.agent = _make_tier(num_blocks=4)

    def test_lookup_empty_tier(self):
        assert lookup_and_wait(self.tier, [key(1)]) == [LookupResult.MISS]

    def test_store_and_lookup(self):
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        assert lookup_and_wait(self.tier, [key(1)]) == [LookupResult.HIT]

    def test_lookup_unrelated_key_returns_false(self):
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        drain(self.tier)
        assert lookup_and_wait(self.tier, [key(999)]) == [LookupResult.MISS]

    def test_store_then_load_roundtrip(self):
        self.tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
        results = drain(self.tier)
        assert results[0].success

        self.tier.submit_load(make_job(2, [key(1), key(2)], [0, 1]))
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success

    def test_multiple_jobs_tracked_independently(self):
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        self.tier.submit_store(make_job(2, [key(2)], [1]))
        results = drain(self.tier)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_failed_transfer_reported(self):
        self.agent.check_xfer_state = lambda h: "ERR"
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(self.tier)
        assert len(results) == 1
        assert not results[0].success

    def test_failed_load_marks_verdict_negative(self):
        """Regression for the failed-load livelock on the obj tier: a
        cached HIT must not survive a failed load of the same key. On the
        failed promotion the tier marks the verdict False from
        get_finished_jobs() (drained here) on the scheduler thread; otherwise
        the scheduler would re-issue the same doomed promotion every step for
        the life of the request. The mark is served from cache with no
        re-probe, so even though the mock object is still 'present' the SAME
        request now resolves to MISS."""
        ctx = ReqContext(req_id="obj-livelock")
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(self.tier))
        # Cache a positive verdict: the object is present, so lookup is a HIT.
        assert lookup_and_wait(self.tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

        # The promotion the HIT triggered fails.
        self.agent.check_xfer_state = lambda h: "ERR"
        self.tier.submit_load(make_job(2, [key(1)], [0]))
        results = drain(self.tier)
        assert len(results) == 1 and not results[0].success

        # After the failed promotion the SAME request's lookup must resolve to
        # MISS (verdict marked False) instead of serving the stale HIT — even
        # though the object itself is still present in the mock store.
        assert lookup_and_wait(self.tier, [key(1)], ctx=ctx) == [LookupResult.MISS]

    def test_pending_transfer_not_returned_until_done(self):
        # First poll returns PROC; second poll returns DONE.
        call_count = [0]
        original = self.agent.check_xfer_state

        def delayed(h):
            call_count[0] += 1
            return "PROC" if call_count[0] == 1 else original(h)

        self.agent.check_xfer_state = delayed

        self.tier.submit_store(make_job(1, [key(1)], [0]))
        assert list(self.tier.get_finished_jobs()) == []
        results = list(self.tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].success

    def test_drain_jobs_polls_until_transfers_complete(self):
        """drain_jobs must keep polling check_xfer_state until every
        in-flight transfer finishes. A buggy implementation that only
        polled once would return with _transfers still populated.
        """
        call_count = [0]
        original = self.agent.check_xfer_state

        def delayed(h):
            call_count[0] += 1
            # Stay in PROC for the first 2 polls, then DONE.
            return "PROC" if call_count[0] < 3 else original(h)

        self.agent.check_xfer_state = delayed

        self.tier.submit_store(make_job(1, [key(1)], [0]))
        assert self.tier._transfers  # in flight

        self.tier.drain_jobs()

        assert not self.tier._transfers  # fully drained
        assert call_count[0] >= 3  # polled past the initial PROC responses
        # Result is buffered for the next get_finished_jobs() call.
        results = list(self.tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].success


class TestMockObjTierMultiBlock:
    def test_store_multiple_blocks(self):
        tier, _ = _make_tier(num_blocks=8)
        keys = [key(i) for i in range(8)]
        tier.submit_store(make_job(1, keys, list(range(8))))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert lookup_and_wait(tier, keys) == [LookupResult.HIT] * 8

    def test_partial_block_lookup(self):
        tier, _ = _make_tier(num_blocks=4)
        tier.submit_store(make_job(1, [key(0), key(1)], [0, 1]))
        drain(tier)
        assert lookup_and_wait(tier, [key(0), key(1), key(2)]) == [
            LookupResult.HIT,
            LookupResult.HIT,
            LookupResult.MISS,
        ]


class TestMockObjTierFailures:
    def test_lookup_exception_returns_false(self):
        tier, agent = _make_tier(num_blocks=4)
        agent.query_memory = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("backend error")
        )
        assert lookup_and_wait(tier, [key(1)]) == [LookupResult.MISS]

    def test_submit_store_register_memory_failure_reported_in_get_finished(self):
        tier, agent = _make_tier(num_blocks=4)
        agent.register_memory = lambda *a, **k: None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = list(tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 1
        assert not results[0].success

    def test_submit_load_register_memory_failure_reported_in_get_finished(self):
        tier, agent = _make_tier(num_blocks=4)
        agent.register_memory = lambda *a, **k: None
        tier.submit_load(make_job(2, [key(1)], [0]))
        results = list(tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 2
        assert not results[0].success

    def test_submit_load_exception_clears_job_keys(self):
        tier, _ = _make_tier(num_blocks=4)
        tier._submit_transfer = MagicMock(side_effect=RuntimeError("submit failed"))

        with pytest.raises(RuntimeError, match="submit failed"):
            tier.submit_load(make_job(2, [key(1)], [0]))

        assert tier._load_job_keys == {}

    def test_submit_store_make_prepped_xfer_failure_reported_in_get_finished(self):
        tier, agent = _make_tier(num_blocks=4)
        agent.make_prepped_xfer = lambda *a, **k: None
        tier.submit_store(make_job(3, [key(1)], [0]))
        results = list(tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 3
        assert not results[0].success

    def test_failure_and_success_both_returned_by_get_finished(self):
        # One job fails at submission, another succeeds in flight.
        tier, agent = _make_tier(num_blocks=4)
        original_register = agent.register_memory
        call_count = [0]

        def register_once_fail(*a, **k):
            call_count[0] += 1
            return None if call_count[0] == 1 else original_register(*a, **k)

        agent.register_memory = register_once_fail

        tier.submit_store(make_job(1, [key(1)], [0]))  # fails immediately
        tier.submit_store(make_job(2, [key(2)], [1]))  # succeeds
        results = drain(tier)
        assert len(results) == 2
        by_id = {r.job_id: r for r in results}
        assert not by_id[1].success
        assert by_id[2].success

    def test_release_xfer_failure_retries_without_losing_result(self, monkeypatch):
        tier, agent = _make_tier(num_blocks=4)
        agent.check_xfer_state = MagicMock(side_effect=RuntimeError("poll failed"))
        release_xfer = MagicMock(
            side_effect=[RuntimeError("transfer is still active"), None]
        )
        monkeypatch.setattr(agent, "release_xfer_handle", release_xfer)

        tier.submit_store(make_job(1, [key(1)], [0]))

        # The transfer handle could not be released safely, so the job must
        # remain tracked and must not be finalized yet.
        assert list(tier.get_finished_jobs()) == []
        assert 1 in tier._transfers

        # Cleanup is retried without polling again or changing the failure
        # verdict. The completion is then returned exactly once.
        results = list(tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 1
        assert not results[0].success
        assert agent.check_xfer_state.call_count == 2
        assert release_xfer.call_count == 2
        assert not tier._transfers
        assert list(tier.get_finished_jobs()) == []

    @pytest.mark.parametrize(
        "cleanup_method", ["release_dlist_handle", "deregister_memory"]
    )
    def test_post_transfer_cleanup_failure_does_not_lose_result(
        self, monkeypatch, cleanup_method
    ):
        tier, agent = _make_tier(num_blocks=4)
        monkeypatch.setattr(
            agent,
            cleanup_method,
            MagicMock(side_effect=RuntimeError("cleanup failed")),
        )

        tier.submit_store(make_job(1, [key(1)], [0]))
        results = list(tier.get_finished_jobs())

        assert len(results) == 1
        assert results[0].job_id == 1
        assert results[0].success
        assert not tier._transfers
        assert list(tier.get_finished_jobs()) == []

    def test_xfer_cleanup_retry_finalizes_parent_job_and_primary_pin(self, monkeypatch):
        num_blocks = 4
        tensor = torch.zeros((num_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
        primary_kv_view = memoryview(tensor.numpy())
        mmap_region = MagicMock()
        mmap_region.create_kv_memoryview.return_value = primary_kv_view
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_blocks, mmap_region=mmap_region
        )
        obj_tier, agent = _make_tier(
            num_blocks=num_blocks, primary_kv_view=primary_kv_view
        )
        manager = TieringOffloadingManager(
            primary_tier=primary_tier, secondary_tiers=[obj_tier]
        )

        keys = [key(1)]
        primary_result = primary_tier.prepare_store(keys, _CTX)
        assert primary_result is not None
        primary_tier.complete_store(keys, _CTX, success=True)
        job = manager.create_store_job(keys, _CTX)
        obj_tier.submit_store(job)

        block = primary_tier._policy.get(keys[0])
        assert block is not None
        assert block.ref_cnt == 1
        assert len(manager._transfer_jobs) == 1

        agent.check_xfer_state = MagicMock(side_effect=RuntimeError("poll failed"))
        release_xfer = MagicMock(
            side_effect=[RuntimeError("transfer is still active"), None]
        )
        monkeypatch.setattr(agent, "release_xfer_handle", release_xfer)
        schedule_context = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())

        manager.on_schedule_end(schedule_context)

        assert len(manager._transfer_jobs) == 1
        assert block.ref_cnt == 1
        assert len(obj_tier._transfers) == 1
        assert manager.has_pending_work()

        manager.on_schedule_end(schedule_context)

        assert manager._transfer_jobs == {}
        assert block.ref_cnt == 0
        assert obj_tier._transfers == {}
        assert not manager.has_pending_work()
        assert agent.check_xfer_state.call_count == 2
        assert release_xfer.call_count == 2


class TestMockObjTierShutdown:
    def test_shutdown_clears_in_flight_transfers(self):
        tier, agent = _make_tier(num_blocks=4)
        # Keep transfer in flight by never completing it
        agent.check_xfer_state = lambda h: "PROC"
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert len(tier._transfers) == 1
        tier.shutdown()
        assert len(tier._transfers) == 0
        assert tier._dram_prepped_handle is None
        assert tier._primary_reg is None

    def test_shutdown_idempotent(self):
        tier, _ = _make_tier(num_blocks=4)
        tier.shutdown()
        tier.shutdown()  # must not raise


class TestObjTierKVEvents:
    def setup_method(self):
        self.tier, self.agent = _make_tier(
            offloading_spec=_make_events_spec(
                enable_kv_cache_events=True,
                self_describing_kv_events=True,
            ),
            enable_kv_events=True,
            locality="REMOTE",
        )

    def test_successful_store_emits_stored_event(self):
        """A completed store transfer emits one stored event with the job's keys."""
        keys = [key(1), key(2)]
        self.tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(self.tier))

        event_iter = iter(self.tier.take_events())
        event = next(event_iter)
        assert self.tier.events == []
        assert event.keys == keys
        assert event.medium == Medium.STORAGE
        assert event.locality is Locality.REMOTE
        assert not event.removed
        assert event.req_context is _CTX
        assert self.tier._store_job_contexts == {}
        event_iter.close()
        # take_events drains the buffer.
        assert list(self.tier.take_events()) == []

    @pytest.mark.parametrize(
        ("locality", "expected"),
        [(None, None), ("LOCAL", Locality.LOCAL)],
    )
    def test_store_event_uses_configured_locality(self, locality, expected):
        locality_config = {} if locality is None else {"locality": locality}
        tier, _ = _make_tier(
            offloading_spec=_make_events_spec(enable_kv_cache_events=True),
            enable_kv_events=True,
            **locality_config,
        )
        try:
            tier.submit_store(make_job(1, [key(1)], [0]))
            assert all(r.success for r in drain(tier))

            events = list(tier.take_events())
            assert len(events) == 1
            assert events[0].locality is expected
            assert events[0].req_context is None
        finally:
            tier.shutdown()

    def test_mixed_job_results_emit_event_only_for_successful_job(self):
        """With a failed and a successful store job resolving in the same
        poll, exactly one event is emitted and its keys belong to the
        successful job."""
        original = self.agent.check_xfer_state
        self.agent.check_xfer_state = lambda h: "ERR" if h._id == 0 else original(h)
        self.tier.submit_store(make_job(1, [key(1)], [0]))  # handle 0: fails
        self.tier.submit_store(make_job(2, [key(2)], [1]))  # handle 1: succeeds
        results = drain(self.tier)
        by_id = {r.job_id: r for r in results}
        assert not by_id[1].success
        assert by_id[2].success

        events = list(self.tier.take_events())
        assert len(events) == 1
        assert events[0].keys == [key(2)]
        assert self.tier._store_job_keys == {}

    def test_load_job_emits_no_event(self):
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        list(self.tier.take_events())

        self.tier.submit_load(make_job(2, [key(1)], [0]))
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        assert list(self.tier.take_events()) == []

    def test_failed_transfer_emits_no_event(self):
        self.agent.check_xfer_state = lambda h: "ERR"
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(self.tier)
        assert not results[0].success
        assert list(self.tier.take_events()) == []
        assert self.tier._store_job_keys == {}
        assert self.tier._store_job_contexts == {}

    def test_submission_failure_emits_no_event(self):
        self.agent.make_prepped_xfer = lambda *a, **k: None
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        results = list(self.tier.get_finished_jobs())
        assert not results[0].success
        assert list(self.tier.take_events()) == []
        assert self.tier._store_job_keys == {}
        assert self.tier._store_job_contexts == {}

    def test_events_disabled_by_default(self):
        tier, _ = _make_tier()
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert tier.events is None
        assert tier._store_job_keys == {}
        assert list(tier.take_events()) == []

    def test_events_require_global_kv_events_flag(self):
        """Tier-level opt-in alone is not enough; the global flag gates events."""
        tier, _ = _make_tier(
            offloading_spec=_make_events_spec(enable_kv_cache_events=False),
            enable_kv_events=True,
        )
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert tier.events is None
        assert tier._store_job_keys == {}
        assert list(tier.take_events()) == []


class TestObjStoreConfig:
    def test_explicit_credentials_included(self):
        cfg = ObjStoreConfig(
            bucket="b",
            endpoint_override="ep",
            access_key="ak",
            secret_key="sk",
        )
        params = cfg.to_nixl_params()
        assert params["access_key"] == "ak"
        assert params["secret_key"] == "sk"

    def test_credentials_omitted_when_empty(self):
        cfg = ObjStoreConfig(bucket="b", endpoint_override="ep")
        params = cfg.to_nixl_params()
        assert "access_key" not in params
        assert "secret_key" not in params
        assert "session_token" not in params
        assert "region" not in params
        assert params["bucket"] == "b"
        assert params["endpoint_override"] == "ep"

    def test_session_token_and_region_included(self):
        cfg = ObjStoreConfig(
            bucket="b",
            endpoint_override="ep",
            access_key="ak",
            secret_key="sk",
            session_token="tok",
            region="us-east-1",
        )
        params = cfg.to_nixl_params()
        assert params["session_token"] == "tok"
        assert params["region"] == "us-east-1"

    def test_ca_bundle_included_when_set(self):
        cfg = ObjStoreConfig(
            bucket="b",
            endpoint_override="ep",
            ca_bundle="/path/to/ca.pem",
        )
        params = cfg.to_nixl_params()
        assert params["ca_bundle"] == "/path/to/ca.pem"
        assert "access_key" not in params


def test_obj_tier_replicated_layout_collapses_mapper_identity():
    """TP=2 and TP=4 replicated configs share the obj FileMapper namespace."""
    tp2_spec = SimpleNamespace(
        config=_make_offloading_config(
            False, tp_size=2, world_size=2, rank=1, replicated_layout=True
        ),
        kv_events_config=OffloadingKVEventsConfig(
            enable_kv_cache_events=False,
            self_describing_kv_events=False,
        ),
    )
    tp4_spec = SimpleNamespace(
        config=_make_offloading_config(
            False, tp_size=4, world_size=4, rank=3, replicated_layout=True
        ),
        kv_events_config=OffloadingKVEventsConfig(
            enable_kv_cache_events=False,
            self_describing_kv_events=False,
        ),
    )
    tp2_tier, _ = _make_tier(offloading_spec=tp2_spec)
    tp4_tier, _ = _make_tier(offloading_spec=tp4_spec)
    try:
        assert tp2_tier._file_mapper.base_path == tp4_tier._file_mapper.base_path
        assert tp2_tier._file_mapper.rank == 0
        assert tp4_tier._file_mapper.rank == 0
        assert tp2_tier._file_mapper.get_run_config() == (
            tp4_tier._file_mapper.get_run_config()
        )
    finally:
        tp2_tier.shutdown()
        tp4_tier.shutdown()
