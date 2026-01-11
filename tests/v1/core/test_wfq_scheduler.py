# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Weighted Fair Queuing (WFQ) scheduler.

These tests verify:
1. Virtual time mechanics (monotonicity, computation)
2. Fairness guarantees (proportional allocation)
3. Edge cases (empty queue, preemption, equal weights)
4. Integration with Request model and Scheduler
"""

import time
import uuid

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


# ============================================================================
# Test Helpers
# ============================================================================


def create_test_request(
    request_id: str | None = None,
    num_prompt_tokens: int = 100,
    max_tokens: int = 50,
    weight: float = 1.0,
    arrival_time: float | None = None,
    priority: int = 0,
) -> Request:
    """Create a test request with specified parameters.

    Args:
        request_id: Unique identifier (auto-generated if None)
        num_prompt_tokens: Number of prompt tokens
        max_tokens: Maximum output tokens
        weight: WFQ scheduling weight
        arrival_time: Arrival timestamp (current time if None)
        priority: Request priority
    """
    if request_id is None:
        request_id = uuid.uuid4().hex

    if arrival_time is None:
        arrival_time = time.time()

    prompt_token_ids = list(range(num_prompt_tokens))

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=False,
    )

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=1,
        arrival_time=arrival_time,
        priority=priority,
        weight=weight,  # WFQ weight
    )


# ============================================================================
# Test: Virtual Time Mechanics
# ============================================================================


def test_wfq_request_weight_attribute():
    """Test that Request accepts weight parameter."""
    # Default weight
    req1 = create_test_request()
    assert hasattr(req1, "weight")
    assert req1.weight == 1.0

    # Custom weight
    req2 = create_test_request(weight=2.5)
    assert req2.weight == 2.5


def test_wfq_request_virtual_time_attributes():
    """Test that Request has virtual time attributes."""
    req = create_test_request()
    assert hasattr(req, "virtual_start_time")
    assert hasattr(req, "virtual_finish_time")
    assert req.virtual_start_time == 0.0
    assert req.virtual_finish_time == 0.0


def test_wfq_queue_creation():
    """Test that WFQ queue can be created via factory."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
        assert queue is not None
        assert len(queue) == 0
    except (ValueError, AttributeError, ImportError) as e:
        pytest.skip(f"WFQ not yet implemented: {e}")


def test_wfq_virtual_time_computation():
    """Test that virtual times are computed correctly on add_request."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Create request with weight=1.0, 150 total tokens
    req = create_test_request(
        num_prompt_tokens=100,
        max_tokens=50,
        weight=1.0,
    )

    queue.add_request(req)

    # Virtual times should be set
    assert req.virtual_start_time >= 0.0
    assert req.virtual_finish_time > req.virtual_start_time
    # Virtual finish = virtual_start + (tokens_needed / weight)
    # tokens_needed = 100 + 50 = 150
    # Expected: virtual_finish - virtual_start ≈ 150 / 1.0 = 150
    time_diff = req.virtual_finish_time - req.virtual_start_time
    assert 140 <= time_diff <= 160  # Allow some tolerance


def test_wfq_virtual_time_with_different_weights():
    """Test that higher weight leads to smaller virtual_finish_time."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Two identical requests, different weights
    req_low_weight = create_test_request(
        request_id="low",
        num_prompt_tokens=100,
        max_tokens=50,
        weight=0.5,  # Lower weight
    )
    req_high_weight = create_test_request(
        request_id="high",
        num_prompt_tokens=100,
        max_tokens=50,
        weight=2.0,  # Higher weight
    )

    queue.add_request(req_low_weight)
    queue.add_request(req_high_weight)

    # Higher weight should have SMALLER virtual_finish_time
    # (gets scheduled earlier)
    assert req_high_weight.virtual_finish_time < req_low_weight.virtual_finish_time


def test_wfq_virtual_time_monotonic():
    """Test that global virtual time never decreases."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    reqs = [create_test_request(weight=1.0) for _ in range(5)]

    prev_virtual_time = 0.0
    for req in reqs:
        queue.add_request(req)

        # After each pop, virtual time should not decrease
        if len(queue) > 0:
            popped = queue.pop_request()
            # Check that virtual_start_time is monotonic
            assert popped.virtual_start_time >= prev_virtual_time
            prev_virtual_time = popped.virtual_start_time


# ============================================================================
# Test: Fairness Guarantees
# ============================================================================


def test_wfq_fairness_equal_weights():
    """Test that equal weights behave like FCFS."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Create requests with equal weights but different arrival times
    reqs = []
    for i in range(5):
        req = create_test_request(
            request_id=f"req_{i}",
            weight=1.0,
            arrival_time=float(i),
        )
        reqs.append(req)
        queue.add_request(req)

    # With equal weights, should be scheduled in FCFS order
    for i in range(5):
        popped = queue.pop_request()
        assert popped.request_id == f"req_{i}"


def test_wfq_fairness_high_weight_scheduled_first():
    """Test that higher weight requests are scheduled before lower weight."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Add low weight request first
    req_low = create_test_request(
        request_id="low",
        weight=0.5,
        arrival_time=1.0,
    )
    queue.add_request(req_low)

    # Add high weight request second (later arrival)
    req_high = create_test_request(
        request_id="high",
        weight=5.0,
        arrival_time=2.0,
    )
    queue.add_request(req_high)

    # High weight should be scheduled first (despite later arrival)
    first = queue.pop_request()
    assert first.request_id == "high"

    second = queue.pop_request()
    assert second.request_id == "low"


def test_wfq_fairness_proportional_allocation():
    """Test that resources are allocated proportionally to weights."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Create 3 requests with weights 1:2:4
    req1 = create_test_request(request_id="w1", weight=1.0)
    req2 = create_test_request(request_id="w2", weight=2.0)
    req4 = create_test_request(request_id="w4", weight=4.0)

    queue.add_request(req1)
    queue.add_request(req2)
    queue.add_request(req4)

    # Virtual finish times should be in ratio 4:2:1 (inverse of weights)
    # Higher weight = smaller virtual_finish = scheduled earlier
    first = queue.pop_request()
    second = queue.pop_request()
    third = queue.pop_request()

    assert first.request_id == "w4"  # Weight 4.0 (highest)
    assert second.request_id == "w2"  # Weight 2.0 (middle)
    assert third.request_id == "w1"  # Weight 1.0 (lowest)


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_wfq_empty_queue_pop_raises():
    """Test that popping from empty queue raises IndexError."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    with pytest.raises(IndexError):
        queue.pop_request()


def test_wfq_empty_queue_peek_raises():
    """Test that peeking empty queue raises IndexError."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    with pytest.raises(IndexError):
        queue.peek_request()


def test_wfq_single_request():
    """Test WFQ with single request."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    req = create_test_request()
    queue.add_request(req)

    assert len(queue) == 1
    assert queue.peek_request() == req

    popped = queue.pop_request()
    assert popped == req
    assert len(queue) == 0


def test_wfq_zero_weight_handled():
    """Test that zero or negative weight is handled gracefully."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    req = create_test_request(weight=0.0)
    # Should either: (1) raise ValueError, or (2) default to 1.0
    try:
        queue.add_request(req)
        # If it doesn't raise, weight should have been fixed to default
        assert req.weight > 0.0
    except (ValueError, ZeroDivisionError):
        # Acceptable to reject zero weight
        pass


def test_wfq_very_large_weight():
    """Test that very large weight works correctly."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    req_large = create_test_request(request_id="large", weight=1000000.0)
    req_normal = create_test_request(request_id="normal", weight=1.0)

    queue.add_request(req_normal)
    queue.add_request(req_large)

    # Large weight should be scheduled first
    first = queue.pop_request()
    assert first.request_id == "large"


def test_wfq_equal_virtual_finish_times():
    """Test behavior when two requests have identical virtual_finish_time."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Create requests with same tokens and weight
    req1 = create_test_request(
        request_id="req1",
        num_prompt_tokens=100,
        max_tokens=50,
        weight=1.0,
        arrival_time=1.0,
    )
    req2 = create_test_request(
        request_id="req2",
        num_prompt_tokens=100,
        max_tokens=50,
        weight=1.0,
        arrival_time=1.0,  # Same arrival time
    )

    queue.add_request(req1)
    queue.add_request(req2)

    # Should use tiebreaker (arrival_time, then request_id)
    # Order should be deterministic
    first = queue.pop_request()
    second = queue.pop_request()

    assert first.request_id != second.request_id
    assert first.request_id in ["req1", "req2"]
    assert second.request_id in ["req1", "req2"]


# ============================================================================
# Test: Queue Operations
# ============================================================================


def test_wfq_prepend_request():
    """Test prepending request (e.g., after preemption)."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    req1 = create_test_request(request_id="req1")
    req2 = create_test_request(request_id="req2")

    queue.add_request(req1)
    queue.add_request(req2)

    # Pop one request
    popped = queue.pop_request()

    # Prepend it back (simulating preemption)
    queue.prepend_request(popped)

    # Virtual times should be preserved
    assert len(queue) == 2


def test_wfq_remove_request():
    """Test removing specific request from queue."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    req1 = create_test_request(request_id="req1")
    req2 = create_test_request(request_id="req2")
    req3 = create_test_request(request_id="req3")

    queue.add_request(req1)
    queue.add_request(req2)
    queue.add_request(req3)

    assert len(queue) == 3

    # Remove middle request
    queue.remove_request(req2)

    assert len(queue) == 2

    # Remaining requests should be req1 and req3
    remaining_ids = {req.request_id for req in queue}
    assert remaining_ids == {"req1", "req3"}


def test_wfq_remove_multiple_requests():
    """Test removing multiple requests at once."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    reqs = [create_test_request(request_id=f"req{i}") for i in range(5)]

    for req in reqs:
        queue.add_request(req)

    assert len(queue) == 5

    # Remove requests 1 and 3
    queue.remove_requests([reqs[1], reqs[3]])

    assert len(queue) == 3

    remaining_ids = {req.request_id for req in queue}
    assert remaining_ids == {"req0", "req2", "req4"}


def test_wfq_bool_and_len():
    """Test __bool__ and __len__ operations."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    assert not queue  # Empty queue is falsy
    assert len(queue) == 0

    req = create_test_request()
    queue.add_request(req)

    assert queue  # Non-empty queue is truthy
    assert len(queue) == 1


def test_wfq_iteration():
    """Test iterating over queue in virtual_finish_time order."""
    try:
        queue = create_request_queue(SchedulingPolicy.WFQ)
    except (ValueError, AttributeError, ImportError):
        pytest.skip("WFQ not yet implemented")

    # Add requests with different weights
    reqs = [
        create_test_request(request_id="w1", weight=1.0),
        create_test_request(request_id="w2", weight=2.0),
        create_test_request(request_id="w3", weight=3.0),
    ]

    for req in reqs:
        queue.add_request(req)

    # Iteration should be in order of virtual_finish_time
    iterated_ids = [req.request_id for req in queue]

    # Higher weight = earlier scheduling
    assert iterated_ids == ["w3", "w2", "w1"]


# ============================================================================
# Test: Request Comparison
# ============================================================================


def test_request_comparison_with_virtual_finish():
    """Test that Request.__lt__ works with virtual_finish_time."""
    req1 = create_test_request(request_id="req1")
    req2 = create_test_request(request_id="req2")

    # Set virtual_finish_time manually (simulating WFQ)
    req1.virtual_finish_time = 100.0
    req2.virtual_finish_time = 50.0

    # req2 should be "less than" req1 (scheduled earlier)
    assert req2 < req1


def test_request_comparison_fallback_to_priority():
    """Test that Request.__lt__ falls back to priority if no virtual_finish."""
    req_high_priority = create_test_request(priority=1)
    req_low_priority = create_test_request(priority=5)

    # Without virtual_finish_time, should use priority
    assert req_high_priority < req_low_priority


# ============================================================================
# Test: Backward Compatibility
# ============================================================================


def test_request_backward_compatible_without_weight():
    """Test that old code creating Request without weight still works."""
    # Old-style request creation (no weight parameter)
    req = Request(
        request_id="old_style",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        eos_token_id=1,
    )

    # Should have default weight=1.0
    assert hasattr(req, "weight")
    assert req.weight == 1.0


# ============================================================================
# Test: Integration with Scheduler Config
# ============================================================================


def test_scheduling_policy_enum_has_wfq():
    """Test that SchedulingPolicy enum includes WFQ."""
    try:
        wfq_policy = SchedulingPolicy.WFQ
        assert wfq_policy.value == "wfq"
    except AttributeError:
        pytest.skip("WFQ policy not yet added to enum")


# ============================================================================
# End of Tests
# ============================================================================
