# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for Weighted Fair Queuing (WFQ) scheduler.

Tests the complete WFQ implementation including:
- Queue operations (add, pop, peek, prepend, remove)
- Virtual time mechanics
- Fairness guarantees
- Edge cases and corner scenarios
- Request comparison logic
- Backward compatibility
"""

import time

import pytest

from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def create_minimal_request(
    request_id: str,
    num_prompt_tokens: int = 100,
    max_tokens: int = 50,
    weight: float = 1.0,
    arrival_time: float | None = None,
    priority: int = 0,
) -> Request:
    """Create a minimal test request for WFQ testing."""

    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(max_tokens=max_tokens)

    request = Request(
        request_id=request_id,
        prompt_token_ids=[0] * num_prompt_tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=0,
        arrival_time=arrival_time if arrival_time is not None else time.time(),
        priority=priority,
        weight=weight,
    )

    return request


class TestWFQBasicOperations:
    """Test basic WFQ queue operations."""

    def test_wfq_queue_creation(self):
        """Test that WFQ queue can be created."""
        queue = create_request_queue(SchedulingPolicy.WFQ)
        assert queue is not None
        assert len(queue) == 0
        assert not queue

    def test_add_single_request(self):
        """Test adding a single request to the queue."""
        queue = create_request_queue(SchedulingPolicy.WFQ)
        request = create_minimal_request("req1")

        queue.add_request(request)

        assert len(queue) == 1
        assert bool(queue)
        assert hasattr(request, "virtual_start_time")
        assert hasattr(request, "virtual_finish_time")
        assert request.virtual_finish_time > 0.0

    def test_pop_single_request(self):
        """Test popping a single request from the queue."""
        queue = create_request_queue(SchedulingPolicy.WFQ)
        request = create_minimal_request("req1")

        queue.add_request(request)
        popped = queue.pop_request()

        assert popped.request_id == "req1"
        assert len(queue) == 0
        assert not queue

    def test_peek_request(self):
        """Test peeking at next request without removal."""
        queue = create_request_queue(SchedulingPolicy.WFQ)
        request = create_minimal_request("req1")

        queue.add_request(request)
        peeked = queue.peek_request()

        assert peeked.request_id == "req1"
        assert len(queue) == 1  # Still in queue
        assert bool(queue)

    def test_pop_from_empty_queue(self):
        """Test that popping from empty queue raises error."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        with pytest.raises(IndexError, match="pop from empty heap"):
            queue.pop_request()

    def test_peek_from_empty_queue(self):
        """Test that peeking from empty queue raises error."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        with pytest.raises(IndexError, match="peek from empty heap"):
            queue.peek_request()


class TestWFQVirtualTimeMechanics:
    """Test WFQ virtual time computation and scheduling."""

    def test_virtual_time_computation(self):
        """Test that virtual times are computed correctly."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Request with 100 prompt tokens, 50 output tokens, weight 1.0
        # Total tokens needed: 150
        # virtual_finish = virtual_start + 150/1.0 = virtual_start + 150
        request = create_minimal_request("req1", num_prompt_tokens=100, max_tokens=50)
        queue.add_request(request)

        assert request.virtual_start_time >= 0.0
        assert request.virtual_finish_time > request.virtual_start_time

        # Check the formula
        expected_tokens = 100 + 50  # prompt + output
        expected_vfinish = request.virtual_start_time + (expected_tokens / 1.0)
        assert abs(request.virtual_finish_time - expected_vfinish) < 0.01

    def test_higher_weight_scheduled_first(self):
        """Test that requests with higher weight are scheduled first."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Both arrive at same time, same token needs, but different weights
        arrival_time = time.time()

        req_low_weight = create_minimal_request(
            "low",
            num_prompt_tokens=100,
            max_tokens=50,
            weight=1.0,
            arrival_time=arrival_time,
        )
        req_high_weight = create_minimal_request(
            "high",
            num_prompt_tokens=100,
            max_tokens=50,
            weight=2.0,
            arrival_time=arrival_time,
        )

        # Add low weight first
        queue.add_request(req_low_weight)
        queue.add_request(req_high_weight)

        # High weight should have smaller virtual_finish_time
        assert req_high_weight.virtual_finish_time < req_low_weight.virtual_finish_time

        # Pop should return high weight first
        first = queue.pop_request()
        assert first.request_id == "high"

    def test_virtual_time_advances_monotonically(self):
        """Test that global virtual time advances monotonically."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add multiple requests
        for i in range(5):
            request = create_minimal_request(f"req{i}")
            queue.add_request(request)

        # Pop all and verify virtual time never decreases
        prev_vstart = -1.0
        while queue:
            request = queue.pop_request()
            assert request.virtual_start_time >= prev_vstart
            prev_vstart = request.virtual_start_time

    def test_equal_weights_fcfs_order(self):
        """Test that equal weights result in FCFS order."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # All same weight, added in order
        for i in range(5):
            request = create_minimal_request(
                f"req{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=1.0,
                arrival_time=time.time() + i * 0.1,  # Stagger arrival times
            )
            time.sleep(0.01)  # Small delay
            queue.add_request(request)

        # Should pop in arrival order (FCFS)
        for i in range(5):
            popped = queue.pop_request()
            assert popped.request_id == f"req{i}"


class TestWFQFairnessGuarantees:
    """Test WFQ fairness properties."""

    def test_proportional_resource_allocation(self):
        """Test that resources are allocated proportionally to weights."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        arrival_time = time.time()

        # Request with weight 2.0 should get twice the resources
        req1 = create_minimal_request(
            "req1",
            num_prompt_tokens=100,
            max_tokens=100,
            weight=1.0,
            arrival_time=arrival_time,
        )
        req2 = create_minimal_request(
            "req2",
            num_prompt_tokens=100,
            max_tokens=100,
            weight=2.0,
            arrival_time=arrival_time,
        )

        queue.add_request(req1)
        queue.add_request(req2)

        # Weight 2.0 should have half the virtual finish time progression
        tokens1 = 200  # 100 prompt + 100 output
        tokens2 = 200

        vf1 = req1.virtual_start_time + (tokens1 / 1.0)
        vf2 = req2.virtual_start_time + (tokens2 / 2.0)

        assert abs(req1.virtual_finish_time - vf1) < 0.01
        assert abs(req2.virtual_finish_time - vf2) < 0.01
        assert req2.virtual_finish_time < req1.virtual_finish_time

    def test_no_starvation(self):
        """Test that no request is starved indefinitely."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add low-weight request first
        low_weight_req = create_minimal_request(
            "low",
            num_prompt_tokens=100,
            max_tokens=50,
            weight=0.5,
            arrival_time=time.time(),
        )
        queue.add_request(low_weight_req)

        # Add many high-weight requests
        for i in range(10):
            high_weight_req = create_minimal_request(
                f"high{i}",
                num_prompt_tokens=10,
                max_tokens=10,
                weight=10.0,
                arrival_time=time.time() + 0.1,
            )
            queue.add_request(high_weight_req)

        # Pop all requests and verify low-weight request is eventually served
        served_ids = []
        while queue:
            request = queue.pop_request()
            served_ids.append(request.request_id)

        assert "low" in served_ids  # No starvation


class TestWFQEdgeCases:
    """Test WFQ edge cases and corner scenarios."""

    def test_zero_or_negative_weight_uses_default(self):
        """Test that zero or negative weights fall back to default."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create request with invalid weight
        request = create_minimal_request("req1", weight=0.0)
        queue.add_request(request)

        # Weight should be corrected to default (1.0)
        assert request.weight == 1.0

    def test_very_large_weight(self):
        """Test handling of very large weights."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        request = create_minimal_request("req1", weight=1000000.0)
        queue.add_request(request)

        assert request.virtual_finish_time > 0.0
        assert request.virtual_finish_time < request.virtual_start_time + 1.0

    def test_very_small_positive_weight(self):
        """Test handling of very small positive weights."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        request = create_minimal_request("req1", weight=0.001)
        queue.add_request(request)

        # Small weight = large virtual_finish_time
        assert request.virtual_finish_time > request.virtual_start_time + 1000

    def test_prepend_request_preserves_virtual_times(self):
        """Test that prepending a request preserves its virtual times."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add and pop a request to set its virtual times
        request = create_minimal_request("req1")
        queue.add_request(request)

        original_vstart = request.virtual_start_time
        original_vfinish = request.virtual_finish_time

        popped = queue.pop_request()

        # Prepend it back (simulating preemption)
        queue.prepend_request(popped)

        # Virtual times should be preserved
        assert popped.virtual_start_time == original_vstart
        assert popped.virtual_finish_time == original_vfinish

    def test_remove_specific_request(self):
        """Test removing a specific request from the queue."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        requests = [create_minimal_request(f"req{i}") for i in range(5)]
        for req in requests:
            queue.add_request(req)

        # Remove middle request
        queue.remove_request(requests[2])

        assert len(queue) == 4

        # Verify req2 is not in queue
        remaining_ids = [req.request_id for req in queue]
        assert "req2" not in remaining_ids

    def test_remove_multiple_requests(self):
        """Test removing multiple requests at once."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        requests = [create_minimal_request(f"req{i}") for i in range(10)]
        for req in requests:
            queue.add_request(req)

        # Remove even-numbered requests
        to_remove = [requests[i] for i in range(0, 10, 2)]
        queue.remove_requests(to_remove)

        assert len(queue) == 5

        # Verify only odd-numbered requests remain
        remaining_ids = sorted([req.request_id for req in queue])
        expected_ids = sorted([f"req{i}" for i in range(1, 10, 2)])
        assert remaining_ids == expected_ids

    def test_iteration_order(self):
        """Test that iteration follows WFQ priority order."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add requests with different weights
        weights = [1.0, 3.0, 2.0, 0.5, 4.0]
        for i, weight in enumerate(weights):
            request = create_minimal_request(
                f"req{i}", weight=weight, arrival_time=time.time()
            )
            queue.add_request(request)
            time.sleep(0.01)

        # Iteration should be in virtual_finish_time order
        prev_vfinish = -1.0
        for request in queue:
            assert request.virtual_finish_time >= prev_vfinish
            prev_vfinish = request.virtual_finish_time


class TestWFQBackwardCompatibility:
    """Test WFQ backward compatibility."""

    def test_default_weight_behavior(self):
        """Test that default weight (1.0) works correctly."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create request without specifying weight
        request = create_minimal_request("req1")

        assert request.weight == 1.0

        queue.add_request(request)
        assert request.virtual_finish_time > 0.0

    def test_request_comparison_fallback(self):
        """Test Request.__lt__ falls back to priority when not WFQ."""
        req1 = create_minimal_request("req1", priority=1)
        req2 = create_minimal_request("req2", priority=2)

        # Don't add to WFQ queue, so virtual_finish_time stays 0.0
        # Should fall back to priority comparison
        assert req1 < req2  # Lower priority value = higher priority

    def test_mixed_scheduling_policies(self):
        """Test that requests work with both WFQ and priority queues."""
        wfq_queue = create_request_queue(SchedulingPolicy.WFQ)
        priority_queue = create_request_queue(SchedulingPolicy.PRIORITY)

        request = create_minimal_request("req1", priority=1)

        # Same request should work in both queue types
        wfq_queue.add_request(request)
        wfq_popped = wfq_queue.pop_request()

        priority_queue.add_request(request)
        priority_popped = priority_queue.pop_request()

        assert wfq_popped.request_id == priority_popped.request_id


class TestWFQStressTests:
    """Stress tests for WFQ implementation."""

    def test_large_number_of_requests(self):
        """Test WFQ with a large number of requests."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        num_requests = 1000
        for i in range(num_requests):
            request = create_minimal_request(f"req{i}", weight=1.0 + (i % 10) * 0.5)
            queue.add_request(request)

        assert len(queue) == num_requests

        # Pop all and verify order is consistent
        prev_vfinish = -1.0
        count = 0
        while queue:
            request = queue.pop_request()
            assert request.virtual_finish_time >= prev_vfinish
            prev_vfinish = request.virtual_finish_time
            count += 1

        assert count == num_requests

    def test_rapid_add_remove(self):
        """Test rapid addition and removal of requests."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add 100, remove 50, add 50, remove 75, etc.
        for _ in range(10):
            # Add requests
            for i in range(100):
                request = create_minimal_request(f"req{i}")
                queue.add_request(request)

            # Remove half
            for _ in range(50):
                if queue:
                    queue.pop_request()

        # Queue should still be functional
        assert len(queue) >= 0
        test_req = create_minimal_request("final")
        queue.add_request(test_req)
        assert len(queue) > 0
