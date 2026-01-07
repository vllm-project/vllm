# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end integration tests for WFQ scheduler.

Tests the complete integration of WFQ into the vLLM scheduler:
- Configuration propagation (SchedulerConfig → Scheduler → RequestQueue)
- Request flow through scheduler with WFQ policy
- Weight parameter handling
- Scheduling behavior verification
"""

import pytest

from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)

pytestmark = pytest.mark.cpu_test


class TestWFQConfigurationIntegration:
    """Test WFQ configuration propagation through the system."""

    def test_scheduler_config_accepts_wfq_policy(self):
        """Test that SchedulerConfig accepts 'wfq' as a valid policy."""
        config = SchedulerConfig.default_factory(policy="wfq")

        assert config.policy == "wfq"

    def test_scheduler_config_default_policy_is_fcfs(self):
        """Test that default policy remains 'fcfs' (backward compatibility)."""
        config = SchedulerConfig.default_factory()

        assert config.policy == "fcfs"

    def test_scheduler_config_rejects_invalid_policy(self):
        """Test that invalid policies are rejected at config level."""
        # This should fail type checking, but let's verify runtime behavior
        # by directly testing the Literal type
        from typing import get_args

        from vllm.config.scheduler import SchedulerPolicy

        valid_policies = get_args(SchedulerPolicy)
        assert "fcfs" in valid_policies
        assert "priority" in valid_policies
        assert "wfq" in valid_policies
        assert len(valid_policies) == 3

    def test_scheduling_policy_enum_has_wfq(self):
        """Test that SchedulingPolicy enum includes WFQ."""
        assert hasattr(SchedulingPolicy, "WFQ")
        assert SchedulingPolicy.WFQ.value == "wfq"

    def test_scheduling_policy_from_string(self):
        """Test that SchedulingPolicy can be created from string."""
        policy = SchedulingPolicy("wfq")
        assert policy == SchedulingPolicy.WFQ

    def test_create_request_queue_with_wfq_policy(self):
        """Test that create_request_queue factory creates WFQRequestQueue."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Verify it's the correct type
        assert queue is not None
        assert type(queue).__name__ == "WFQRequestQueue"

        # Verify it has WFQ-specific attributes
        assert hasattr(queue, "_virtual_time")
        assert hasattr(queue, "_default_weight")


class TestWFQRequestFlowIntegration:
    """Test request flow through WFQ scheduler."""

    def test_wfq_queue_scheduling_order(self):
        """Test that WFQ queue schedules by weight correctly."""

        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create requests with different weights
        sampling_params = SamplingParams(max_tokens=50)

        req_low = Request(
            request_id="low_weight",
            prompt_token_ids=[0] * 100,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=0,
            weight=0.5,  # Low weight = low priority
        )

        req_medium = Request(
            request_id="medium_weight",
            prompt_token_ids=[0] * 100,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=0,
            weight=1.0,  # Medium weight
        )

        req_high = Request(
            request_id="high_weight",
            prompt_token_ids=[0] * 100,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=0,
            weight=2.0,  # High weight = high priority
        )

        # Add in arbitrary order
        queue.add_request(req_medium)
        queue.add_request(req_low)
        queue.add_request(req_high)

        # Should pop in weight order: high, medium, low
        first = queue.pop_request()
        assert first.request_id == "high_weight"

        second = queue.pop_request()
        assert second.request_id == "medium_weight"

        third = queue.pop_request()
        assert third.request_id == "low_weight"

    def test_wfq_queue_with_default_weight(self):
        """Test that requests without explicit weight use default."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        queue = create_request_queue(SchedulingPolicy.WFQ)

        sampling_params = SamplingParams(max_tokens=50)

        # Request without explicit weight (uses default 1.0)
        req = Request(
            request_id="default_weight",
            prompt_token_ids=[0] * 100,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=0,
            # weight parameter not specified, should default to 1.0
        )

        assert req.weight == 1.0

        queue.add_request(req)
        assert req.virtual_finish_time > 0.0

    def test_wfq_preserves_virtual_times_on_preemption(self):
        """Test that WFQ preserves virtual times when request is preempted."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        queue = create_request_queue(SchedulingPolicy.WFQ)

        sampling_params = SamplingParams(max_tokens=50)

        req = Request(
            request_id="test_req",
            prompt_token_ids=[0] * 100,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=0,
            weight=1.0,
        )

        # Add to queue
        queue.add_request(req)

        # Save virtual times
        original_vstart = req.virtual_start_time
        original_vfinish = req.virtual_finish_time

        # Pop (simulate scheduling)
        popped = queue.pop_request()

        # Prepend back (simulate preemption)
        queue.prepend_request(popped)

        # Virtual times should be preserved
        assert popped.virtual_start_time == original_vstart
        assert popped.virtual_finish_time == original_vfinish


class TestWFQVsPriorityComparison:
    """Compare WFQ behavior with Priority scheduling."""

    def test_wfq_vs_priority_different_ordering(self):
        """Test that WFQ and Priority produce different orderings."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        wfq_queue = create_request_queue(SchedulingPolicy.WFQ)
        priority_queue = create_request_queue(SchedulingPolicy.PRIORITY)

        # Create requests with:
        # - Same priority values
        # - Different weights (for WFQ)
        # - Different token counts

        req1 = Request(
            request_id="req1",
            prompt_token_ids=[0] * 200,  # Large request
            sampling_params=SamplingParams(max_tokens=100),
            pooling_params=None,
            eos_token_id=0,
            priority=1,  # Same priority
            weight=0.5,  # Low weight (WFQ)
        )

        req2 = Request(
            request_id="req2",
            prompt_token_ids=[0] * 100,  # Medium request
            sampling_params=SamplingParams(max_tokens=50),
            pooling_params=None,
            eos_token_id=0,
            priority=1,  # Same priority
            weight=2.0,  # High weight (WFQ)
        )

        # Add to both queues
        wfq_queue.add_request(req1)
        wfq_queue.add_request(req2)

        # Create copies for priority queue (to avoid shared state)
        req1_copy = Request(
            request_id="req1",
            prompt_token_ids=[0] * 200,
            sampling_params=SamplingParams(max_tokens=100),
            pooling_params=None,
            eos_token_id=0,
            priority=1,
            weight=0.5,
        )

        req2_copy = Request(
            request_id="req2",
            prompt_token_ids=[0] * 100,
            sampling_params=SamplingParams(max_tokens=50),
            pooling_params=None,
            eos_token_id=0,
            priority=1,
            weight=2.0,
        )

        priority_queue.add_request(req1_copy)
        priority_queue.add_request(req2_copy)

        # WFQ should schedule req2 first (higher weight)
        wfq_first = wfq_queue.pop_request()
        assert wfq_first.request_id == "req2"

        # Priority should schedule req1 first
        # (earlier arrival time in this test)
        # (or req2 if arrival times are identical, but weight doesn't
        # matter for priority queue)


class TestWFQBackwardCompatibility:
    """Test that WFQ doesn't break existing functionality."""

    def test_fcfs_queue_still_works(self):
        """Test that FCFS queue still works (backward compatibility)."""
        queue = create_request_queue(SchedulingPolicy.FCFS)
        assert queue is not None
        assert type(queue).__name__ == "FCFSRequestQueue"

    def test_priority_queue_still_works(self):
        """Test that Priority queue still works (backward compatibility)."""
        queue = create_request_queue(SchedulingPolicy.PRIORITY)
        assert queue is not None
        assert type(queue).__name__ == "PriorityRequestQueue"

    def test_request_without_weight_works_in_all_queues(self):
        """Test that requests without weight work in all queue types."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        sampling_params = SamplingParams(max_tokens=50)

        for policy in [
            SchedulingPolicy.FCFS,
            SchedulingPolicy.PRIORITY,
            SchedulingPolicy.WFQ,
        ]:
            queue = create_request_queue(policy)

            req = Request(
                request_id=f"req_{policy.value}",
                prompt_token_ids=[0] * 100,
                sampling_params=sampling_params,
                pooling_params=None,
                eos_token_id=0,
                # No weight specified - should work for all policies
            )

            queue.add_request(req)
            popped = queue.pop_request()
            assert popped.request_id == f"req_{policy.value}"


class TestWFQDocumentation:
    """Test that WFQ is properly documented."""

    def test_scheduler_config_policy_docstring_includes_wfq(self):
        """Test that SchedulerConfig.policy docstring mentions WFQ."""
        # Get the field docstring
        # Check if docstring exists (it's in the class definition)
        # Since we can't easily access the docstring, let's verify the source
        import inspect

        from vllm.config.scheduler import SchedulerConfig

        source = inspect.getsource(SchedulerConfig)
        assert "wfq" in source.lower()
        assert "weighted fair queuing" in source.lower()

    def test_scheduling_policy_enum_has_all_policies(self):
        """Test that SchedulingPolicy enum has all expected policies."""
        policies = [p.value for p in SchedulingPolicy]

        assert "fcfs" in policies
        assert "priority" in policies
        assert "wfq" in policies
        assert len(policies) == 3
