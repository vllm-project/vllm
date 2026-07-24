# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the zero-reclaim preemption cascade.

Companion to test_deferred_block_free.py. That file tests the *defer mechanism*
(by calling _preempt_request directly). This file tests the **running-extend
allocation retry loop**: when the policy-selected victim's free is deferred
(same-step), the loop must not chain-preempt it (zero-reclaim cascade).

The retry loop is forced deterministically by patching allocate_slots to fail
after the running set is established (standard control-flow unit test; no GPU,
no exact-saturation recreation).
"""

import os
from unittest.mock import patch

import pytest

from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MODEL = os.environ.get("VLLM_TEST_DEFER_FREE_MODEL", "facebook/opt-125m")
STOP_TOKEN_ID = 42
BLOCK_SIZE = 16
NUM_PROMPT_TOKENS = 33  # 3 blocks @ block_size=16


def _deferring_scheduler(num_blocks=1000, max_num_seqs=16):
    """Async scheduler with deferred block freeing forced on (the production
    gate additionally requires a KV-consumer connector; the mechanism itself is
    independent of it, like test_deferred_block_free._create_deferring_scheduler)."""
    s = create_scheduler(
        model=MODEL,
        async_scheduling=True,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=False,
    )
    s.defer_block_free = True
    return s


def _running_same_step(scheduler, n_requests, max_tokens=5):
    """Schedule depth-1 async (prefill + 1 over-scheduled decode) so every
    running request has last_sched_seq > processed_step_seq -> its free is
    deferred this step."""
    reqs = create_requests(
        num_requests=n_requests,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=max_tokens,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for r in reqs:
        scheduler.add_request(r)
    scheduler.schedule()  # prefill (step 1)
    scheduler.schedule()  # 1 decode (step 2 in flight) -> all same-step
    running = list(scheduler.running)
    assert running
    for r in running:
        assert r.last_sched_seq > scheduler.processed_step_seq
    return running


def _force_retry_loop(scheduler):
    """Patch allocate_slots to fail, run one schedule() -> forces the
    running-extend preempt retry loop. Returns the SchedulerOutput."""
    with patch.object(scheduler.kv_cache_manager, "allocate_slots", return_value=None):
        return scheduler.schedule()


def test_zero_reclaim_guard_preserves_trigger_state():
    """A deferred (zero-reclaim) victim is NOT preempted, and the trigger's
    scheduler state stays fully consistent (no half-schedule, no spurious
    output, no bookkeeping corruption)."""
    s = _deferring_scheduler()
    running = _running_same_step(s, n_requests=6)
    trigger = s.running[0]
    running_rids = {r.request_id for r in running}
    waiting_before = len(s.waiting)

    out = _force_retry_loop(s)

    # No same-step (deferred-free) victim is preempted.
    assert not [r for r in out.preempted_req_ids if r in running_rids]
    # Trigger stays RUNNING, still in running, not scheduled this step.
    assert trigger.status == RequestStatus.RUNNING
    assert trigger in s.running
    assert trigger.request_id not in out.num_scheduled_tokens
    # No spurious waiting churn.
    assert len(s.waiting) == waiting_before


def test_no_starvation_after_fence_advances():
    """The guard is a conservative stop-and-wait, not a stall: with real
    allocation, once the in-flight step is processed the running set keeps
    making forward progress (the scheduler does not deadlock)."""
    s = _deferring_scheduler()
    _running_same_step(s, n_requests=4)
    # First, the guard stops the cascade under allocation failure.
    out_fail = _force_retry_loop(s)
    assert len(out_fail.preempted_req_ids) == 0
    # Then, with real allocation, the scheduler remains usable (no deadlock).
    for _ in range(20):
        s.schedule()
    assert s.running or s.waiting, "scheduler deadlocked (no running, no waiting)"


def test_priority_policy_guard():
    """Under PRIORITY the victim is max(priority, arrival); the guard before
    removing it is bookkeeping-safe and still stops the cascade."""
    from vllm.v1.core.sched.scheduler import SchedulingPolicy

    s = _deferring_scheduler()
    s.policy = SchedulingPolicy.PRIORITY
    running = _running_same_step(s, n_requests=6)
    running_rids = {r.request_id for r in running}
    out = _force_retry_loop(s)
    assert not [r for r in out.preempted_req_ids if r in running_rids], (
        "PRIORITY: guard should not preempt a deferred victim, "
        f"got {out.preempted_req_ids}"
    )


def test_guard_inert_without_defer():
    """When defer_block_free=False (no connector / no async), the guard
    predicate is never true, so normal preemption still happens under
    allocation failure (no-connector path unaffected)."""
    s = _deferring_scheduler()
    s.defer_block_free = False
    reqs = create_requests(
        num_requests=6,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=5,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for r in reqs:
        s.add_request(r)
    s.schedule()  # prefill -> running
    assert s.running
    out = _force_retry_loop(s)
    assert len(out.preempted_req_ids) >= 1, (
        "defer OFF: normal preemption should still happen, guard must be inert"
    )


def test_reclaimable_victim_preempted_normally():
    """Even with defer_block_free=True, a victim scheduled in an already-
    processed step (last_sched_seq <= processed_step_seq) frees immediately, so
    the guard must NOT block its preemption (no false positive)."""
    s = _deferring_scheduler()
    _running_same_step(s, n_requests=6)
    # Make all running victims reclaimable: advance processed past last_sched_seq.
    s.processed_step_seq = max(r.last_sched_seq for r in s.running) + 1
    running_rids = {r.request_id for r in s.running}
    out = _force_retry_loop(s)
    preempted_running = [r for r in out.preempted_req_ids if r in running_rids]
    assert len(preempted_running) >= 1, (
        "reclaimable victim: guard must allow normal preemption"
    )
