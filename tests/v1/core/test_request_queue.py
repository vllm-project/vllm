# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct, isolated unit tests for RequestQueue implementations.

These exercise FCFSRequestQueue and PriorityRequestQueue on their own,
without going through the full Scheduler/engine setup that
test_scheduler.py and test_priority_scheduler_random.py use. In
particular, this regression-tests PriorityRequestQueue.__iter__, which
was changed from copying the heap and popping every element to a single
sorted() call (same order, one fewer O(n) copy).
"""

import random

from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)

from .test_scheduler import create_requests_with_priority


def test_create_request_queue():
    assert isinstance(create_request_queue(SchedulingPolicy.FCFS), FCFSRequestQueue)
    assert isinstance(
        create_request_queue(SchedulingPolicy.PRIORITY), PriorityRequestQueue
    )


def test_fcfs_order():
    queue = FCFSRequestQueue()
    requests = create_requests_with_priority(3, priorities=[0, 0, 0])
    for request in requests:
        queue.add_request(request)

    assert list(queue) == requests
    assert queue.pop_request() is requests[0]
    assert queue.pop_request() is requests[1]
    assert queue.pop_request() is requests[2]
    assert not queue


def test_priority_order_matches_priority_then_arrival():
    queue = PriorityRequestQueue()
    # Deliberately added out of order: priority order should be
    # req_1 (priority 0), req_2 (priority 1), req_0 (priority 2).
    requests = create_requests_with_priority(
        3, priorities=[2, 0, 1], arrival_times=[0.0, 1.0, 2.0]
    )
    for request in requests:
        queue.add_request(request)

    ordered = list(queue)
    assert [r.request_id for r in ordered] == ["1", "2", "0"]
    # peek/pop should agree with iteration order.
    assert queue.peek_request() is requests[1]
    assert queue.pop_request() is requests[1]


def test_priority_arrival_time_tiebreaker():
    # Same priority: earlier arrival_time goes first.
    queue = PriorityRequestQueue()
    requests = create_requests_with_priority(
        3, priorities=[5, 5, 5], arrival_times=[2.0, 0.0, 1.0]
    )
    for request in requests:
        queue.add_request(request)

    assert [r.request_id for r in queue] == ["1", "2", "0"]


def test_priority_iter_does_not_mutate_queue():
    """__iter__ must not drain or reorder the underlying heap -- it's a
    read-only view. (The previous implementation copied the heap first
    specifically to guarantee this; sorted() also never mutates its
    argument, so the guarantee still holds, but it's worth asserting
    explicitly since it's the property callers actually depend on.)"""
    queue = PriorityRequestQueue()
    requests = create_requests_with_priority(5, priorities=[3, 1, 4, 1, 5])
    for request in requests:
        queue.add_request(request)

    before = list(queue)
    first_pass = list(queue)
    second_pass = list(queue)
    after = list(queue)

    assert before == first_pass == second_pass == after
    assert len(queue) == 5


def test_priority_iter_matches_sorted_with_many_ties():
    """Regression test for the __iter__ fix: with many requests sharing
    the same priority AND the same arrival_time, ordering falls through
    to Request.__lt__'s final tiebreakers (request_id, then object
    identity). Iteration order must still be self-consistent with the
    queue's own total order on every call, including before and after
    unrelated pops -- this is the exact edge case that would reveal a
    mismatch if sorted() and repeated heappop() could ever disagree."""
    random.seed(0)
    queue = PriorityRequestQueue()
    # Many requests tied on priority and arrival_time; only request_id
    # (and finally object identity) can break the tie.
    requests = create_requests_with_priority(
        30, priorities=[2] * 30, arrival_times=[1.5] * 30
    )
    shuffled = requests[:]
    random.shuffle(shuffled)
    for request in shuffled:
        queue.add_request(request)

    ordered = list(queue)
    # Ties on priority/arrival_time fall through to request_id, which is
    # a distinct, sortable string per request here, so the final order
    # must be fully determined by it.
    assert [r.request_id for r in ordered] == sorted(r.request_id for r in requests)

    # Must still agree after mutating the queue (pop a few, then re-check
    # the rest are still in a valid, self-consistent total order).
    queue.pop_request()
    queue.pop_request()
    remaining = list(queue)
    assert remaining == sorted(remaining)
    assert len(remaining) == 28


def test_priority_remove_requests():
    queue = PriorityRequestQueue()
    requests = create_requests_with_priority(4, priorities=[0, 1, 2, 3])
    for request in requests:
        queue.add_request(request)

    queue.remove_requests([requests[1], requests[2]])
    assert [r.request_id for r in queue] == ["0", "3"]