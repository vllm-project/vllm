# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlowPrefill adaptive sub-chunk preemption.

FlowPrefill is on vLLM's Q2 2026 roadmap:
  "Address known scheduler issues — avoid excessive preemption,
   prefill HoL blocking"

Reference: FlowPrefill paper (arxiv Feb 2026)

Test coverage:
  - Disabled path (preemption_granularity=None) — zero overhead, unchanged.
  - PrefillCheckpointState dataclass construction and field access.
  - CudaEventPool acquire/release lifecycle.
  - FlowPrefillMixin initialisation and property.
  - Checkpoint insertion decision logic (_fp_maybe_checkpoint).
  - Resume logic (_fp_try_resume_suspended) with mocked CUDA events.
  - Full scheduling loop (_fp_schedule) — checkpoint and non-checkpoint paths.
  - No CPU synchronisation guarantee: query() is non-blocking.
"""

from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.scheduler import (
    CudaEventPool,
    FlowPrefillMixin,
    PrefillCheckpointState,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_event(query_result: bool = True) -> MagicMock:
    """Return a mock that mimics torch.cuda.Event with a non-blocking query."""
    event = MagicMock()
    event.query.return_value = query_result
    event.record = MagicMock()
    return event


def _make_mock_scheduler(
    granularity: Optional[int] = 128,
    decode_threshold: int = 1,
    running_count: int = 0,
) -> FlowPrefillMixin:
    """Return a minimal FlowPrefillMixin instance with mocked config."""

    class _FakeCfg:
        preemption_granularity = granularity
        preemption_decode_threshold = decode_threshold

    class _FakeScheduler(FlowPrefillMixin):
        def __init__(self) -> None:
            self.scheduler_config = _FakeCfg()
            self.running = [object() for _ in range(running_count)]
            self._init_flowprefill()

    return _FakeScheduler()


def _make_fake_request(
    request_id: str,
    num_computed_tokens: int = 0,
    num_remaining_tokens: int = 512,
    num_allocated_blocks: int = 4,
) -> MagicMock:
    req = MagicMock()
    req.request_id = request_id
    req.num_computed_tokens = num_computed_tokens
    req.num_remaining_tokens = num_remaining_tokens
    req.num_allocated_blocks = num_allocated_blocks
    return req


# ---------------------------------------------------------------------------
# PrefillCheckpointState tests
# ---------------------------------------------------------------------------


def test_prefill_checkpoint_state_fields():
    """PrefillCheckpointState stores all fields and timestamps correctly."""
    event = _make_mock_event()
    before = time.perf_counter_ns()
    state = PrefillCheckpointState(
        request_id="req-1",
        checkpoint_pos=128,
        allocated_kv_blocks_count=8,
        cuda_event=event,
    )
    after = time.perf_counter_ns()

    assert state.request_id == "req-1"
    assert state.checkpoint_pos == 128
    assert state.allocated_kv_blocks_count == 8
    assert state.cuda_event is event
    assert before <= state.created_at_ns <= after


def test_prefill_checkpoint_state_none_event():
    """PrefillCheckpointState accepts None as cuda_event (CPU/test env)."""
    state = PrefillCheckpointState(
        request_id="req-cpu",
        checkpoint_pos=64,
        allocated_kv_blocks_count=2,
        cuda_event=None,
    )
    assert state.cuda_event is None


# ---------------------------------------------------------------------------
# CudaEventPool tests
# ---------------------------------------------------------------------------


def test_event_pool_acquire_and_release():
    """Pool tracks used/free sets correctly across acquire/release cycles."""
    pool = CudaEventPool(size=4)
    assert len(pool._free) == 4
    assert len(pool._used) == 0

    e0 = pool.acquire("r0")
    assert e0 is not None
    assert len(pool._free) == 3
    assert "r0" in pool._used

    e1 = pool.acquire("r1")
    assert e1 is not None
    assert e0 is not e1

    pool.release("r0")
    assert len(pool._free) == 3  # 2 never-acquired + r0 returned
    assert "r0" not in pool._used

    pool.release("r1")
    assert len(pool._free) == 4  # all 4 events back in pool
    assert len(pool._used) == 0


def test_event_pool_exhaustion_returns_none():
    """Pool returns None when all events are in use (no blocking)."""
    pool = CudaEventPool(size=2)
    pool.acquire("r0")
    pool.acquire("r1")
    assert pool._free == []
    result = pool.acquire("r2")
    assert result is None


def test_event_pool_release_unknown_id_is_noop():
    """Releasing an unknown request ID does not raise or corrupt state."""
    pool = CudaEventPool(size=2)
    pool.release("does-not-exist")  # must not raise
    assert len(pool._free) == 2


# ---------------------------------------------------------------------------
# FlowPrefillMixin — disabled path (preemption_granularity=None)
# ---------------------------------------------------------------------------


def test_flowprefill_disabled_by_default():
    """When preemption_granularity=None, flowprefill_enabled is False."""
    sched = _make_mock_scheduler(granularity=None)
    assert not sched.flowprefill_enabled


def test_flowprefill_disabled_checkpoint_returns_none():
    """_fp_maybe_checkpoint returns None when FlowPrefill is disabled."""
    sched = _make_mock_scheduler(granularity=None)
    result = sched._fp_maybe_checkpoint(
        request_id="r0",
        current_pos=0,
        remaining_tokens=1024,
        kv_blocks_count=4,
    )
    assert result is None


def test_flowprefill_disabled_resume_returns_empty():
    """_fp_try_resume_suspended returns [] when FlowPrefill is disabled."""
    sched = _make_mock_scheduler(granularity=None)
    result = sched._fp_try_resume_suspended(token_budget=8192)
    assert result == []


# ---------------------------------------------------------------------------
# FlowPrefillMixin — enabled path
# ---------------------------------------------------------------------------


def test_flowprefill_enabled_when_granularity_set():
    """flowprefill_enabled is True when preemption_granularity is set."""
    sched = _make_mock_scheduler(granularity=128)
    assert sched.flowprefill_enabled
    assert sched._fp_granularity == 128
    assert sched._fp_decode_threshold == 1


def test_flowprefill_no_checkpoint_when_no_decode_pressure():
    """Checkpoint is skipped when decode queue is below threshold."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=2, running_count=1
    )
    result = sched._fp_maybe_checkpoint(
        request_id="r0",
        current_pos=0,
        remaining_tokens=1024,
        kv_blocks_count=4,
    )
    assert result is None


def test_flowprefill_no_checkpoint_when_remainder_fits_in_subchunk():
    """Checkpoint is skipped when remaining_tokens <= granularity."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=2
    )
    result = sched._fp_maybe_checkpoint(
        request_id="r0",
        current_pos=896,
        remaining_tokens=64,
        kv_blocks_count=4,
    )
    assert result is None


def test_flowprefill_checkpoint_inserted_under_decode_pressure():
    """Checkpoint is created when decode queue meets threshold."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=1
    )
    mock_event = _make_mock_event()
    sched._event_pool.acquire = MagicMock(return_value=mock_event)

    result = sched._fp_maybe_checkpoint(
        request_id="r0",
        current_pos=0,
        remaining_tokens=512,
        kv_blocks_count=8,
    )

    assert result is not None
    assert isinstance(result, PrefillCheckpointState)
    assert result.request_id == "r0"
    assert result.checkpoint_pos == 128
    assert result.allocated_kv_blocks_count == 8
    assert result.cuda_event is mock_event
    assert "r0" in sched._suspended_prefills


def test_flowprefill_checkpoint_skipped_when_pool_exhausted():
    """No checkpoint is inserted when the event pool is exhausted."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=1
    )
    sched._event_pool.acquire = MagicMock(return_value=None)

    result = sched._fp_maybe_checkpoint(
        request_id="r0",
        current_pos=0,
        remaining_tokens=512,
        kv_blocks_count=4,
    )
    assert result is None
    assert "r0" not in sched._suspended_prefills


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


def test_fp_resume_when_event_ready_and_no_decode_pressure():
    """Suspended prefill resumes when event fires and decode queue is clear."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=0
    )
    mock_event = _make_mock_event(query_result=True)
    sched._suspended_prefills["r0"] = PrefillCheckpointState(
        request_id="r0",
        checkpoint_pos=128,
        allocated_kv_blocks_count=8,
        cuda_event=mock_event,
    )

    resumed = sched._fp_try_resume_suspended(token_budget=512)

    assert len(resumed) == 1
    req_id, checkpoint, tokens = resumed[0]
    assert req_id == "r0"
    assert tokens == 128
    assert "r0" not in sched._suspended_prefills


def test_fp_no_resume_when_event_not_ready():
    """Suspended prefill stays suspended when CUDA event has not fired."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=0
    )
    mock_event = _make_mock_event(query_result=False)
    sched._suspended_prefills["r0"] = PrefillCheckpointState(
        request_id="r0",
        checkpoint_pos=128,
        allocated_kv_blocks_count=8,
        cuda_event=mock_event,
    )

    resumed = sched._fp_try_resume_suspended(token_budget=512)

    assert resumed == []
    assert "r0" in sched._suspended_prefills


def test_fp_no_resume_when_decode_pressure_high():
    """Suspended prefill stays suspended when decode queue is at/above threshold."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=2, running_count=2
    )
    mock_event = _make_mock_event(query_result=True)
    sched._suspended_prefills["r0"] = PrefillCheckpointState(
        request_id="r0",
        checkpoint_pos=128,
        allocated_kv_blocks_count=8,
        cuda_event=mock_event,
    )

    resumed = sched._fp_try_resume_suspended(token_budget=512)

    assert resumed == []
    assert "r0" in sched._suspended_prefills


# ---------------------------------------------------------------------------
# Full scheduling loop
# ---------------------------------------------------------------------------


def test_fp_schedule_no_decode_pressure_no_checkpoint():
    """_fp_schedule does not insert checkpoints when decode queue is empty."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=0
    )
    requests = [_make_fake_request("r0", num_remaining_tokens=512)]
    scheduled, suspended, budget = sched._fp_schedule(
        waiting_requests=requests,
        token_budget=1024,
        running_requests=[],
    )

    assert len(scheduled) == 1
    assert scheduled[0]["request_id"] == "r0"
    assert scheduled[0].get("is_checkpoint") is False
    assert suspended == []
    assert budget == 1024 - 512


def test_fp_schedule_inserts_checkpoint_under_decode_pressure():
    """_fp_schedule suspends a long prefill and inserts a checkpoint."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=1
    )
    mock_event = _make_mock_event()
    sched._event_pool.acquire = MagicMock(return_value=mock_event)

    requests = [_make_fake_request("r0", num_remaining_tokens=512)]
    scheduled, suspended, budget = sched._fp_schedule(
        waiting_requests=requests,
        token_budget=1024,
        running_requests=[],
    )

    assert len(scheduled) == 1
    assert scheduled[0]["request_id"] == "r0"
    assert scheduled[0]["is_checkpoint"] is True
    assert scheduled[0]["num_tokens"] == 128
    assert "r0" in suspended
    assert budget == 1024 - 128


def test_fp_schedule_event_query_is_nonblocking():
    """CUDA Event.query() must be called without arguments (non-blocking)."""
    sched = _make_mock_scheduler(
        granularity=128, decode_threshold=1, running_count=0
    )
    mock_event = _make_mock_event(query_result=True)
    sched._suspended_prefills["r0"] = PrefillCheckpointState(
        request_id="r0",
        checkpoint_pos=128,
        allocated_kv_blocks_count=8,
        cuda_event=mock_event,
    )

    sched._fp_try_resume_suspended(token_budget=512)

    mock_event.query.assert_called_once_with()
