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
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig
from vllm.v1.kv_offload.tiering.obj.manager import ObjectStoreSecondaryTierManager

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


def _make_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test/model"),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="float16"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            rank=0,
        ),
        use_v2_model_runner=False,
    )


_OFFLOADING_SPEC = SimpleNamespace(
    vllm_config=_make_vllm_config(),
    kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
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
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
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

    def _query_memory(self, queries, mem_type, agent_name):
        return [object() if q[3] in self._stored_obj_keys else None for q in queries]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_events_spec(enable_kv_cache_events: bool) -> SimpleNamespace:
    """Offloading spec stub with an explicit global KV events flag."""
    return SimpleNamespace(
        vllm_config=_make_vllm_config(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=enable_kv_cache_events),
    )


def _make_tier(
    num_blocks: int = 4,
    offloading_spec: SimpleNamespace = _OFFLOADING_SPEC,
    **tier_kwargs,
) -> tuple[ObjectStoreSecondaryTierManager, MockNixlAgent]:
    """Create a tier backed by a fresh MockNixlAgent."""
    mock_agent = MockNixlAgent()
    tensor = torch.zeros((num_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
    view = memoryview(tensor.numpy())
    with (
        patch("vllm.v1.kv_offload.tiering.obj.manager.nixl_agent_config"),
        patch(
            "vllm.v1.kv_offload.tiering.obj.manager.nixl_agent",
            return_value=mock_agent,
        ),
    ):
        tier = ObjectStoreSecondaryTierManager(
            offloading_spec=offloading_spec,
            primary_kv_view=view,
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
            offloading_spec=_make_events_spec(enable_kv_cache_events=True),
            enable_secondary_tier_events=True,
        )

    def test_successful_store_emits_stored_event(self):
        """A completed store transfer emits one stored event with the job's keys."""
        keys = [key(1), key(2)]
        self.tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(self.tier))

        events = list(self.tier.take_events())
        assert len(events) == 1
        assert events[0].keys == keys
        # Literal medium pins the wire contract, not just the constant choice.
        assert events[0].medium == "OBJ"
        assert not events[0].removed
        # take_events drains the buffer.
        assert list(self.tier.take_events()) == []

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

    def test_submission_failure_emits_no_event(self):
        self.agent.make_prepped_xfer = lambda *a, **k: None
        self.tier.submit_store(make_job(1, [key(1)], [0]))
        results = list(self.tier.get_finished_jobs())
        assert not results[0].success
        assert list(self.tier.take_events()) == []
        assert self.tier._store_job_keys == {}

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
            enable_secondary_tier_events=True,
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
