# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PriorityRequestQueue lazy-deletion behaviour.

These tests do not require GPU and can be run with:
    pytest tests/v1/core/test_request_queue.py -v
"""

import time
import uuid

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import (
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def _make_request(priority: int = 0, arrival_time: float | None = None) -> Request:
    """Create a minimal Request for testing."""
    sampling_params = SamplingParams(ignore_eos=False, max_tokens=10)
    return Request(
        request_id=uuid.uuid4().hex,
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
        priority=priority,
        arrival_time=arrival_time if arrival_time is not None else time.time(),
    )


class TestPriorityRequestQueueBasics:
    """Test basic add/pop/peek operations."""

    def test_add_and_pop(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=2)
        r2 = _make_request(priority=1)
        r3 = _make_request(priority=3)
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        # Should pop in priority order (lower first)
        assert q.pop_request() is r2
        assert q.pop_request() is r1
        assert q.pop_request() is r3

    def test_peek(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=2)
        r2 = _make_request(priority=1)
        q.add_request(r1)
        q.add_request(r2)
        assert q.peek_request() is r2
        # peek doesn't remove
        assert len(q) == 2
        assert q.peek_request() is r2

    def test_pop_empty_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_peek_empty_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_bool_and_len(self):
        q = PriorityRequestQueue()
        assert not q
        assert len(q) == 0
        r = _make_request()
        q.add_request(r)
        assert q
        assert len(q) == 1


class TestLazyDeletion:
    """Test lazy deletion for remove_request and remove_requests."""

    def test_remove_single(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        r3 = _make_request(priority=3)
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        q.remove_request(r2)
        assert len(q) == 2
        assert q.pop_request() is r1
        assert q.pop_request() is r3

    def test_remove_head(self):
        """Removing the heap-min element should still work."""
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert len(q) == 1
        assert q.pop_request() is r2

    def test_remove_all(self):
        q = PriorityRequestQueue()
        requests = [_make_request(priority=i) for i in range(5)]
        for r in requests:
            q.add_request(r)
        for r in requests:
            q.remove_request(r)
        assert len(q) == 0
        assert not q
        with pytest.raises(IndexError):
            q.pop_request()

    def test_remove_batch(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        r3 = _make_request(priority=3)
        r4 = _make_request(priority=4)
        for r in [r1, r2, r3, r4]:
            q.add_request(r)
        q.remove_requests([r2, r4])
        assert len(q) == 2
        assert q.pop_request() is r1
        assert q.pop_request() is r3

    def test_remove_batch_with_set(self):
        """remove_requests should accept sets directly."""
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        r3 = _make_request(priority=3)
        for r in [r1, r2, r3]:
            q.add_request(r)
        q.remove_requests({r1, r3})
        assert len(q) == 1
        assert q.pop_request() is r2

    def test_peek_skips_removed(self):
        """peek should skip lazily-deleted entries."""
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert q.peek_request() is r2

    def test_remove_idempotent(self):
        """Removing the same request twice should be safe."""
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        # Second removal via remove_requests should be a no-op
        q.remove_requests([r1])
        assert len(q) == 1
        assert q.pop_request() is r2

    def test_cleanup_threshold(self):
        """Heap should compact when stale entries exceed threshold."""
        q = PriorityRequestQueue()
        requests = [_make_request(priority=i) for i in range(10)]
        for r in requests:
            q.add_request(r)
        # Remove more than half to trigger cleanup
        for r in requests[:6]:
            q.remove_request(r)
        # After cleanup, internal heap should be compact
        assert len(q._heap) == 4
        assert len(q._active_ids) == 4


class TestIteration:
    """Test __iter__ behavior."""

    def test_iter_sorted_order(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=3)
        r2 = _make_request(priority=1)
        r3 = _make_request(priority=2)
        for r in [r1, r2, r3]:
            q.add_request(r)
        result = list(q)
        assert result == [r2, r3, r1]

    def test_iter_skips_removed(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        r3 = _make_request(priority=3)
        for r in [r1, r2, r3]:
            q.add_request(r)
        q.remove_request(r2)
        result = list(q)
        assert result == [r1, r3]

    def test_iter_does_not_modify_queue(self):
        """Iteration should be non-destructive."""
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        q.add_request(r1)
        q.add_request(r2)
        _ = list(q)
        assert len(q) == 2
        assert q.pop_request() is r1


class TestPrepend:
    """Test prepend operations with priority queue."""

    def test_prepend_request(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=2)
        r2 = _make_request(priority=1)
        q.add_request(r1)
        q.prepend_request(r2)
        # prepend_request just calls add_request for priority queue
        assert q.pop_request() is r2

    def test_prepend_requests(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=3)
        r2 = _make_request(priority=1)
        r3 = _make_request(priority=2)
        q.add_request(r1)
        # Create another queue and prepend its contents
        q2 = PriorityRequestQueue()
        q2.add_request(r2)
        q2.add_request(r3)
        q.prepend_requests(q2)
        assert len(q) == 3
        assert q.pop_request() is r2


class TestCreateRequestQueue:
    """Test factory function."""

    def test_priority(self):
        q = create_request_queue(SchedulingPolicy.PRIORITY)
        assert isinstance(q, PriorityRequestQueue)

    def test_fcfs(self):
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue

        q = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(q, FCFSRequestQueue)


class TestRemovalAfterPop:
    """Test that removing a request after it's been popped is a safe no-op."""

    def test_remove_already_popped(self):
        q = PriorityRequestQueue()
        r1 = _make_request(priority=1)
        r2 = _make_request(priority=2)
        q.add_request(r1)
        q.add_request(r2)
        popped = q.pop_request()
        assert popped is r1
        # Removing an already-popped request should be safe
        q.remove_request(r1)
        assert len(q) == 1
        assert q.pop_request() is r2
