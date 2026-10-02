# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the V2 model runner "No free indices" assertion.

The V2 model runner stores per-request state in a fixed-size slab sized to
``max_num_seqs`` (``RequestState.free_indices`` in
``vllm/v1/worker/gpu/states.py``). A slot is occupied from the moment a request
appears in ``scheduled_new_reqs`` (worker ``add_requests``) until it appears in
``finished_req_ids``/``preempted_req_ids`` (worker ``finish_requests``), with a
defensive same-id ``_remove_request`` on re-add. If the scheduler ever lets the
number of slot-holding requests exceed ``max_num_seqs``, ``add_request`` trips::

    assert len(self.free_indices) > 0, "No free indices"

The invariant the scheduler must preserve is: *every request that still holds a
worker slot is counted against the admission limit*. These tests exercise two
ways that invariant can break, by replaying real scheduler outputs through a
faithful model of the worker's slot accounting.
"""

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .test_scheduler import create_scheduler_with_priority
from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class WorkerSlots:
    """Faithful model of the V2 model runner's request-slot accounting.

    Mirrors ``GPUModelRunner.finish_requests`` + ``add_requests`` and
    ``RequestState.add_request``/``remove_request`` (the ``free_indices`` pool).
    ``apply`` raises ``AssertionError("No free indices")`` exactly where the real
    worker would, so a scheduler over-admission surfaces as the same failure.
    """

    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs
        self.occupied: set[str] = set()

    def apply(self, scheduler_output) -> None:
        # finish_requests: free finished and preempted slots first.
        freed = set(scheduler_output.finished_req_ids)
        freed |= set(scheduler_output.preempted_req_ids)
        self.occupied -= freed

        # update_requests: cached reqs must already own a slot, else the
        # worker's req_id_to_index lookup would KeyError.
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            assert req_id in self.occupied, f"cached req {req_id!r} has no worker slot"

        # add_requests: new (and, under V2, resumed) reqs claim a slot. The
        # same-id _remove_request guard frees a stale slot for the same id.
        for new_req in scheduler_output.scheduled_new_reqs:
            self.occupied.discard(new_req.req_id)
            assert len(self.occupied) < self.max_num_reqs, "No free indices"
            self.occupied.add(new_req.req_id)


def _model_runner_output(req_ids: list[str], sampled: list[list[int]]):
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_streaming_pause_does_not_over_admit_worker_slot():
    """A paused resumable session keeps its worker slot; with max_num_seqs=1 a
    new request must not be admitted on top of it.

    Repro of the "No free indices" crash: the session pauses into
    WAITING_FOR_STREAMING_REQ (removed from ``running`` but never reported as
    finished/preempted, so the worker keeps its slot), then a different request
    is admitted into the single slot.
    """
    STOP_TOKEN = 7
    scheduler = create_scheduler(max_num_seqs=1, use_v2_model_runner=True)
    slots = WorkerSlots(max_num_reqs=1)

    # A resumable streaming session.
    (session,) = create_requests(
        num_requests=1,
        num_tokens=4,
        req_ids=["session"],
        stop_token_ids=[STOP_TOKEN],
        max_tokens=16,
    )
    session.resumable = True
    scheduler.add_request(session)

    out = scheduler.schedule()
    slots.apply(out)
    assert [r.req_id for r in out.scheduled_new_reqs] == ["session"]

    # The session emits its stop token and pauses, waiting for the next input
    # chunk. It leaves `running` but is NOT finished/preempted, so the worker
    # still holds its slot.
    scheduler.update_from_output(out, _model_runner_output(["session"], [[STOP_TOKEN]]))
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    # A blocked waiting status lands in skipped_waiting, not the main queue.
    assert session in scheduler.skipped_waiting
    assert scheduler.num_waiting_for_streaming_input == 1

    # A different request arrives while the session is paused.
    (other,) = create_requests(
        num_requests=1, num_tokens=4, req_ids=["other"], max_tokens=16
    )
    scheduler.add_request(other)

    out = scheduler.schedule()
    # Before the fix this admits `other` as a new request while the session's
    # slot is still held -> WorkerSlots.apply raises "No free indices".
    slots.apply(out)

    # After the fix: `other` is deferred until the session resumes or finishes.
    assert "other" not in [r.req_id for r in out.scheduled_new_reqs]
    assert other in scheduler.waiting


def test_reset_prefix_cache_priority_does_not_over_admit_worker_slot():
    """reset_prefix_cache(reset_running_requests=True) force-preempts the running
    request out-of-band; under priority scheduling a higher-priority newcomer
    jumps ahead of the preempted request and would over-admit the single worker
    slot unless the preemption is reported to the worker.

    The fix buffers the force-preemption and reports it in the next scheduler
    output's preempted_req_ids, so the worker frees the slot before the newcomer
    is admitted.
    """
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1, enable_prefix_caching=True, use_v2_model_runner=True
    )
    slots = WorkerSlots(max_num_reqs=1)

    # Low-priority request A (larger priority value == lower priority).
    (req_a,) = create_requests(
        num_requests=1, num_tokens=4, req_ids=["A"], ignore_eos=True, max_tokens=100
    )
    req_a.priority = 10
    scheduler.add_request(req_a)

    out = scheduler.schedule()
    slots.apply(out)
    assert [r.req_id for r in out.scheduled_new_reqs] == ["A"]

    scheduler.update_from_output(out, _model_runner_output(["A"], [[42]]))
    assert req_a.status == RequestStatus.RUNNING
    assert req_a in scheduler.running

    # Force a prefix-cache reset that preempts A out-of-band. A goes back to the
    # waiting queue (PREEMPTED); the preemption is buffered for the next output.
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert req_a.status == RequestStatus.PREEMPTED

    # A higher-priority request B arrives and outranks the preempted A.
    (req_b,) = create_requests(
        num_requests=1, num_tokens=4, req_ids=["B"], ignore_eos=True, max_tokens=100
    )
    req_b.priority = 0
    scheduler.add_request(req_b)

    out = scheduler.schedule()
    # The fix reports A's force-preemption here, so the worker frees A's slot
    # before B is admitted -> no overflow.
    assert "A" in out.preempted_req_ids
    assert "B" in [r.req_id for r in out.scheduled_new_reqs]
    slots.apply(out)
    assert slots.occupied == {"B"}


def test_reset_prefix_cache_same_step_resume_purges_then_re_adds():
    """When a force-preempted request resumes in the SAME step (FCFS, no
    competing request), it appears in both preempted_req_ids and (under V2)
    scheduled_new_reqs. The worker runs finish_requests before add_requests, so
    the request's slot is purged and then cleanly re-added.
    """
    scheduler = create_scheduler(max_num_seqs=1, use_v2_model_runner=True)
    slots = WorkerSlots(max_num_reqs=1)

    (req_a,) = create_requests(
        num_requests=1, num_tokens=4, req_ids=["A"], ignore_eos=True, max_tokens=100
    )
    scheduler.add_request(req_a)

    out = scheduler.schedule()
    slots.apply(out)
    scheduler.update_from_output(out, _model_runner_output(["A"], [[42]]))
    assert req_a in scheduler.running

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert req_a.status == RequestStatus.PREEMPTED
    # Reset invalidates A's computed tokens; it will re-prefill from scratch.
    assert req_a.num_computed_tokens == 0

    out = scheduler.schedule()
    # A is force-preempted AND resumed in this one step.
    assert "A" in out.preempted_req_ids
    assert "A" in [r.req_id for r in out.scheduled_new_reqs]
    # WorkerSlots mirrors finish-before-add: purge frees the slot, re-add fills
    # it. No overflow, A still occupies its single slot afterward.
    slots.apply(out)
    assert slots.occupied == {"A"}
