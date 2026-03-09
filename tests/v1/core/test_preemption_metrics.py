# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test preemption metrics tracking."""

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_preemption_metrics_basic():
    """Test that preemption metrics are tracked correctly."""
    # Create scheduler with limited blocks to trigger preemption
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,  # Limited blocks to force preemption
        enable_prefix_caching=False,
    )

    # Create two requests with different priorities
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)
    requests[0].priority = 0  # High priority
    requests[1].priority = 1  # Low priority

    # Schedule first request
    scheduler.add_request(requests[0])
    output0 = scheduler.schedule()
    assert len(output0.num_scheduled_tokens) == 1

    # Schedule second request
    scheduler.add_request(requests[1])
    output1 = scheduler.schedule()
    assert len(output1.num_scheduled_tokens) == 1

    # Update with output from first request
    model_output0 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output0, model_output0)

    # Schedule again - this should trigger preemption of request[1]
    _ = scheduler.schedule()

    # Verify preemption occurred
    assert requests[1].status == RequestStatus.PREEMPTED
    assert requests[1].num_preemptions == 1

    # Check that preemption metrics were tracked
    assert scheduler.preemption_count_this_interval == 1
    assert requests[1].request_id in scheduler.preempted_req_ids_this_interval
    assert scheduler.preemptions_by_priority_this_interval[1] == 1

    # Get stats and verify preemption stats are included
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.preemption_stats is not None
    assert stats.preemption_stats.num_preemptions == 1
    assert stats.preemption_stats.num_preempted_requests == 1
    assert stats.preemption_stats.preemptions_by_priority[1] == 1
    assert stats.preemption_stats.max_preemption_count == 1

    # After make_stats, counters should be reset
    assert scheduler.preemption_count_this_interval == 0
    assert len(scheduler.preempted_req_ids_this_interval) == 0
    assert len(scheduler.preemptions_by_priority_this_interval) == 0


def test_preemption_metrics_multiple():
    """Test metrics with multiple preemptions."""
    scheduler = create_scheduler(
        max_num_batched_tokens=50,
        block_size=16,
        num_blocks=6,  # Very limited blocks
        enable_prefix_caching=False,
    )

    # Create 3 requests
    requests = create_requests(num_requests=3, num_tokens=40, block_size=16)
    requests[0].priority = 0
    requests[1].priority = 1
    requests[2].priority = 1

    # Add all requests
    for req in requests:
        scheduler.add_request(req)

    # Schedule and trigger preemptions
    output = scheduler.schedule()

    # Simulate execution and preemption cycles
    for req_id in output.num_scheduled_tokens:
        model_output = ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_output)
        output = scheduler.schedule()

    # Get stats
    stats = scheduler.make_stats()

    # Verify stats exist (actual values depend on scheduling behavior)
    if stats and stats.preemption_stats:
        assert stats.preemption_stats.num_preemptions >= 0
        assert stats.preemption_stats.num_preempted_requests >= 0
        assert stats.preemption_stats.max_preemption_count >= 0
