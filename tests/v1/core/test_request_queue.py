# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FCFSRequestQueue and PriorityRequestQueue."""

import pytest

from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Minimal stub: only the fields / methods the queues actually use.
# ---------------------------------------------------------------------------
class _Req:
    """Lightweight stand-in for vllm.v1.request.Request."""

    def __init__(
        self,
        req_id: str,
        priority: int = 0,
        arrival_time: float = 0.0,
    ):
        self.request_id = req_id
        self.priority = priority
        self.arrival_time = arrival_time

    # Heap ordering: lower priority value = higher importance.
    # Ties broken by arrival_time, then request_id.
    def __lt__(self, other: "_Req") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.request_id < other.request_id

    def __repr__(self) -> str:
        return f"Req({self.request_id}, p={self.priority})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reqs(*ids: str, priority: int = 0) -> list[_Req]:
    return [
        _Req(i, priority=priority, arrival_time=float(idx)) for idx, i in enumerate(ids)
    ]


# ===========================================================================
# FCFSRequestQueue
# ===========================================================================


class TestFCFSRequestQueue:
    def test_add_and_pop_fifo_order(self):
        q = FCFSRequestQueue()
        reqs = _make_reqs("a", "b", "c")
        for r in reqs:
            q.add_request(r)
        assert [q.pop_request() for _ in range(3)] == reqs

    def test_bool_and_len(self):
        q = FCFSRequestQueue()
        assert not q
        assert len(q) == 0
        q.add_request(_Req("x"))
        assert q
        assert len(q) == 1

    def test_peek_does_not_remove(self):
        q = FCFSRequestQueue()
        r = _Req("a")
        q.add_request(r)
        assert q.peek_request() is r
        assert len(q) == 1

    def test_peek_empty_raises(self):
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_prepend_request(self):
        q = FCFSRequestQueue()
        reqs = _make_reqs("a", "b")
        for r in reqs:
            q.add_request(r)
        front = _Req("front")
        q.prepend_request(front)
        assert q.pop_request() is front

    def test_prepend_requests(self):
        q = FCFSRequestQueue()
        other = FCFSRequestQueue()
        a, b = _Req("a"), _Req("b")
        other.add_request(a)
        other.add_request(b)
        q.prepend_requests(other)
        # extendleft reverses the order
        assert len(q) == 2

    def test_remove_request(self):
        q = FCFSRequestQueue()
        reqs = _make_reqs("a", "b", "c")
        for r in reqs:
            q.add_request(r)
        q.remove_request(reqs[1])
        remaining = list(q)
        assert reqs[1] not in remaining
        assert len(q) == 2

    def test_remove_requests_bulk(self):
        q = FCFSRequestQueue()
        reqs = _make_reqs("a", "b", "c", "d")
        for r in reqs:
            q.add_request(r)
        q.remove_requests([reqs[0], reqs[2]])
        remaining = list(q)
        assert reqs[0] not in remaining
        assert reqs[2] not in remaining
        assert len(q) == 2

    def test_iter_preserves_order(self):
        q = FCFSRequestQueue()
        reqs = _make_reqs("a", "b", "c")
        for r in reqs:
            q.add_request(r)
        assert list(q) == reqs


# ===========================================================================
# PriorityRequestQueue — basic behaviour
# ===========================================================================


class TestPriorityRequestQueueBasic:
    def test_empty_queue(self):
        q = PriorityRequestQueue()
        assert not q
        assert len(q) == 0

    def test_pop_empty_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_peek_empty_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_single_request(self):
        q = PriorityRequestQueue()
        r = _Req("a", priority=5)
        q.add_request(r)
        assert bool(q)
        assert len(q) == 1
        assert q.peek_request() is r
        assert q.pop_request() is r
        assert not q

    def test_priority_ordering(self):
        """Lower priority value = served first."""
        q = PriorityRequestQueue()
        high = _Req("high", priority=1, arrival_time=1.0)
        low = _Req("low", priority=9, arrival_time=0.0)  # arrived earlier
        q.add_request(low)
        q.add_request(high)
        assert q.pop_request() is high  # priority 1 beats priority 9
        assert q.pop_request() is low

    def test_arrival_time_tiebreak(self):
        q = PriorityRequestQueue()
        first = _Req("first", priority=0, arrival_time=1.0)
        second = _Req("second", priority=0, arrival_time=2.0)
        q.add_request(second)
        q.add_request(first)
        assert q.pop_request() is first
        assert q.pop_request() is second

    def test_iter_priority_order(self):
        q = PriorityRequestQueue()
        reqs = [
            _Req("c", priority=3, arrival_time=0.0),
            _Req("a", priority=1, arrival_time=0.0),
            _Req("b", priority=2, arrival_time=0.0),
        ]
        for r in reqs:
            q.add_request(r)
        ordered = list(q)
        assert ordered[0].request_id == "a"
        assert ordered[1].request_id == "b"
        assert ordered[2].request_id == "c"

    def test_prepend_request_uses_priority(self):
        """prepend_request must respect heap order, not insert at front."""
        q = PriorityRequestQueue()
        low = _Req("low", priority=10)
        high = _Req("high", priority=1)
        q.add_request(low)
        q.prepend_request(high)  # even though "prepended", heap wins
        assert q.pop_request() is high

    def test_prepend_requests_from_queue(self):
        q = PriorityRequestQueue()
        other = PriorityRequestQueue()
        r1 = _Req("r1", priority=5)
        r2 = _Req("r2", priority=2)
        other.add_request(r1)
        other.add_request(r2)
        q.prepend_requests(other)
        assert len(q) == 2
        assert q.pop_request() is r2  # priority 2 first


# ===========================================================================
# PriorityRequestQueue — lazy deletion
# ===========================================================================


class TestPriorityRequestQueueLazyDeletion:
    def test_remove_then_pop_skips_removed(self):
        q = PriorityRequestQueue()
        r1 = _Req("r1", priority=1)
        r2 = _Req("r2", priority=2)
        q.add_request(r1)
        q.add_request(r2)

        q.remove_request(r1)  # lazy-mark r1
        assert len(q) == 1
        assert bool(q)
        assert q.pop_request() is r2  # r1 purged transparently

    def test_remove_then_peek_skips_removed(self):
        q = PriorityRequestQueue()
        r1 = _Req("r1", priority=1)
        r2 = _Req("r2", priority=2)
        q.add_request(r1)
        q.add_request(r2)

        q.remove_request(r1)
        assert q.peek_request() is r2

    def test_remove_all_makes_queue_empty(self):
        q = PriorityRequestQueue()
        reqs = [_Req(str(i), priority=i) for i in range(5)]
        for r in reqs:
            q.add_request(r)

        q.remove_requests(reqs)
        assert not q
        assert len(q) == 0

    def test_remove_requests_bulk(self):
        q = PriorityRequestQueue()
        reqs = [_Req(str(i), priority=i) for i in range(5)]
        for r in reqs:
            q.add_request(r)

        q.remove_requests(reqs[:3])  # remove first 3
        assert len(q) == 2
        remaining = list(q)
        assert set(r.request_id for r in remaining) == {"3", "4"}

    def test_duplicate_remove_is_safe(self):
        """Removing the same request twice must not raise or corrupt state."""
        q = PriorityRequestQueue()
        r = _Req("r", priority=1)
        other = _Req("other", priority=2)
        q.add_request(r)
        q.add_request(other)

        q.remove_request(r)
        q.remove_request(r)  # second remove — already marked, no crash

        assert len(q) == 1
        assert q.pop_request() is other

    def test_remove_nonexistent_does_not_corrupt(self):
        """Marking a request that was never added must not break the queue."""
        q = PriorityRequestQueue()
        real = _Req("real", priority=1)
        ghost = _Req("ghost", priority=0)  # never added
        q.add_request(real)

        q.remove_request(ghost)  # ghost is not in heap
        # The queue must still return `real` correctly.
        assert len(q) == 1
        assert q.pop_request() is real

    def test_iter_excludes_removed(self):
        q = PriorityRequestQueue()
        reqs = [_Req(str(i), priority=i) for i in range(4)]
        for r in reqs:
            q.add_request(r)

        q.remove_request(reqs[1])
        q.remove_request(reqs[3])

        result = list(q)
        assert reqs[1] not in result
        assert reqs[3] not in result
        assert len(result) == 2

    def test_len_and_bool_reflect_lazy_deletions(self):
        q = PriorityRequestQueue()
        reqs = [_Req(str(i), priority=i) for i in range(3)]
        for r in reqs:
            q.add_request(r)

        assert len(q) == 3
        q.remove_request(reqs[0])
        assert len(q) == 2
        assert bool(q)
        q.remove_request(reqs[1])
        q.remove_request(reqs[2])
        assert len(q) == 0
        assert not bool(q)

    def test_interleaved_add_remove(self):
        """Add and remove in interleaved fashion; verify heap stays correct."""
        q = PriorityRequestQueue()
        r1 = _Req("r1", priority=1)
        r2 = _Req("r2", priority=2)
        r3 = _Req("r3", priority=3)

        q.add_request(r3)
        q.add_request(r1)
        q.remove_request(r1)  # remove highest-priority before it's popped
        q.add_request(r2)

        # r1 is gone; expect r2 then r3
        assert q.pop_request() is r2
        assert q.pop_request() is r3
        assert not q

    def test_pop_all_after_partial_removes(self):
        """Exhaust queue with mix of real pops and prior lazy deletes."""
        q = PriorityRequestQueue()
        reqs = [_Req(str(i), priority=i) for i in range(6)]
        for r in reqs:
            q.add_request(r)

        q.remove_requests([reqs[1], reqs[3], reqs[5]])
        expected = [reqs[0], reqs[2], reqs[4]]

        popped = []
        while q:
            popped.append(q.pop_request())
        assert popped == expected


# ===========================================================================
# create_request_queue factory
# ===========================================================================


class TestCreateRequestQueue:
    def test_fcfs_policy(self):
        q = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(q, FCFSRequestQueue)

    def test_priority_policy(self):
        q = create_request_queue(SchedulingPolicy.PRIORITY)
        assert isinstance(q, PriorityRequestQueue)

    def test_unknown_policy_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            create_request_queue("unknown_policy")  # type: ignore[arg-type]
