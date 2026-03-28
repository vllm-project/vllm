# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Scheduler.make_stats() populating num_skipped_waiting_reqs.

These tests need torch (via create_scheduler) so they are tagged cpu_test.
They exercise only the make_stats() → SchedulerStats path; the routing
logic that places requests into skipped_waiting is tested separately
in test_scheduler.py.

Design approach
---------------
We directly manipulate scheduler.waiting and scheduler.skipped_waiting
rather than driving requests through the full add_request() → schedule()
cycle.  This keeps each test focused on a single assertion and avoids
test fragility from unrelated scheduler constraints (token budgets, KV
cache limits, etc.).

The key invariant we are guarding:
  stats.num_waiting_reqs = len(waiting) + len(skipped_waiting)   (combined)
  stats.num_skipped_waiting_reqs = len(skipped_waiting)           (subset)
"""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_stats(scheduler):
    """
    Call make_stats() and assert it returns a non-None result.
    (make_stats returns None when log_stats=False; create_scheduler
    passes log_stats=True so this should never be None.)
    """
    stats = scheduler.make_stats()
    assert stats is not None, (
        "make_stats() returned None — was create_scheduler called with "
        "log_stats=True?"
    )
    return stats


def _add_to_skipped(scheduler, request):
    """
    Place a request directly into skipped_waiting, bypassing the normal
    add_request() routing.  This simulates the state the scheduler reaches
    after a scheduling pass defers a request due to a transient constraint.

    We also register the request in scheduler.requests so finish_requests()
    and other scheduler methods remain consistent.
    """
    scheduler.skipped_waiting.add_request(request)
    scheduler.requests[request.request_id] = request


def _add_to_waiting(scheduler, request):
    """Place a request directly into the main waiting queue."""
    scheduler.waiting.add_request(request)
    scheduler.requests[request.request_id] = request


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_make_stats_empty_queues():
    """
    With no requests queued, both waiting counts must be 0.
    Verifies the baseline / clean-slate behaviour.
    """
    scheduler = create_scheduler()
    stats = _get_stats(scheduler)

    assert stats.num_running_reqs == 0
    assert stats.num_waiting_reqs == 0
    assert stats.num_skipped_waiting_reqs == 0


def test_make_stats_all_in_waiting_queue():
    """
    Requests in the main waiting queue must appear in num_waiting_reqs
    but NOT inflate num_skipped_waiting_reqs.  These are fresh arrivals
    that have never been attempted by the scheduler.
    """
    scheduler = create_scheduler()
    requests = create_requests(num_requests=5)
    for req in requests:
        _add_to_waiting(scheduler, req)

    stats = _get_stats(scheduler)

    assert stats.num_waiting_reqs == 5
    # No deferred requests — the skipped count must be 0.
    assert stats.num_skipped_waiting_reqs == 0


def test_make_stats_all_in_skipped_waiting_queue():
    """
    Requests in skipped_waiting count toward BOTH num_waiting_reqs (for
    backward compatibility) AND num_skipped_waiting_reqs (the new split).
    """
    scheduler = create_scheduler()
    requests = create_requests(num_requests=4)
    for req in requests:
        _add_to_skipped(scheduler, req)

    stats = _get_stats(scheduler)

    # Skipped requests are still "waiting" from the combined metric's view.
    assert stats.num_waiting_reqs == 4
    # All of them are in the deferred sub-population.
    assert stats.num_skipped_waiting_reqs == 4


def test_make_stats_mixed_queues_correct_split():
    """
    The most important case: some requests are freshly queued, others have
    been deferred.  Verify the split is reported correctly.
    """
    scheduler = create_scheduler()

    fresh_requests = create_requests(num_requests=3, req_ids=["f0", "f1", "f2"])
    skipped_requests = create_requests(num_requests=2, req_ids=["s0", "s1"])

    for req in fresh_requests:
        _add_to_waiting(scheduler, req)
    for req in skipped_requests:
        _add_to_skipped(scheduler, req)

    stats = _get_stats(scheduler)

    # Combined count stays correct (backward compat).
    assert stats.num_waiting_reqs == 5  # 3 fresh + 2 skipped

    # New split: only the deferred requests.
    assert stats.num_skipped_waiting_reqs == 2

    # Derived "pure backlog" metric operators can compute.
    pure_backlog = stats.num_waiting_reqs - stats.num_skipped_waiting_reqs
    assert pure_backlog == 3


def test_make_stats_skipped_is_subset_invariant():
    """
    Invariant: num_skipped_waiting_reqs must never exceed num_waiting_reqs.
    This is guaranteed structurally (skipped_waiting ⊆ waiting), but
    we make it explicit so a future refactor doesn't silently break it.
    """
    scheduler = create_scheduler()

    requests = create_requests(num_requests=6, req_ids=[str(i) for i in range(6)])
    for req in requests[:4]:
        _add_to_waiting(scheduler, req)
    for req in requests[4:]:
        _add_to_skipped(scheduler, req)

    stats = _get_stats(scheduler)
    assert stats.num_skipped_waiting_reqs <= stats.num_waiting_reqs


def test_make_stats_running_unaffected_by_skipped():
    """
    Requests in the running list must not affect num_skipped_waiting_reqs.
    This guards against accidental cross-contamination between queues.
    """
    scheduler = create_scheduler()

    # Simulate two running requests by appending directly to the list.
    running_requests = create_requests(num_requests=2, req_ids=["r0", "r1"])
    for req in running_requests:
        scheduler.running.append(req)
        scheduler.requests[req.request_id] = req

    skipped_requests = create_requests(num_requests=3, req_ids=["s0", "s1", "s2"])
    for req in skipped_requests:
        _add_to_skipped(scheduler, req)

    stats = _get_stats(scheduler)

    assert stats.num_running_reqs == 2
    assert stats.num_waiting_reqs == 3
    assert stats.num_skipped_waiting_reqs == 3


def test_make_stats_via_blocked_status_routing():
    """
    End-to-end test of the routing path: add_request() must place requests
    with a blocked waiting status into skipped_waiting, which must then
    appear in num_skipped_waiting_reqs via make_stats().

    This exercises the full path:
      add_request() → _enqueue_waiting_request() → skipped_waiting
      → make_stats() → num_skipped_waiting_reqs

    We use WAITING_FOR_STREAMING_REQ because it is a stable blocked status
    and doesn't require any external connector setup.
    """
    scheduler = create_scheduler()

    # One normal request (goes to waiting).
    normal_req = create_requests(num_requests=1, req_ids=["normal"])[0]
    scheduler.add_request(normal_req)

    # One request whose status is already blocked before being enqueued.
    # The scheduler will route it to skipped_waiting via
    # _enqueue_waiting_request() → _is_blocked_waiting_status() check.
    blocked_req = create_requests(num_requests=1, req_ids=["blocked"])[0]
    blocked_req.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    blocked_req.resumable = True  # required for streaming-input requests
    scheduler.add_request(blocked_req)

    stats = _get_stats(scheduler)

    # Combined count includes both.
    assert stats.num_waiting_reqs == 2

    # Only the blocked request landed in skipped_waiting.
    assert stats.num_skipped_waiting_reqs == 1
