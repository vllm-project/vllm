# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test cases for the scheduler fix that removes the off-by-one error
when scheduling tokens at max_model_len.

These tests verify that:
1. Requests with prompts at max_model_len can be fully processed
2. No infinite loops occur when the final token needs to be scheduled
3. The scheduler correctly handles edge cases around capacity limits
"""

import numpy as np
import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_schedule_max_model_len_prompt():
    """Test that prompts filling max_model_len can be fully scheduled.

    This test verifies the fix for the off-by-one error where prompts
    at max_model_len would get stuck at max_model_len-1 tokens.
    """
    max_model_len = 1024
    token_budget = 512

    scheduler = create_scheduler(
        max_model_len=max_model_len,
        max_num_batched_tokens=token_budget,
    )

    # Create a request with prompt exactly at max_model_len
    requests = create_requests(
        num_requests=1,
        num_tokens=max_model_len,
        max_tokens=1,  # Pooling model scenario
    )
    scheduler.add_request(requests[0])

    # First schedule: should process token_budget tokens
    output1 = scheduler.schedule()
    assert output1.num_scheduled_tokens[requests[0].request_id] == token_budget
    assert requests[0].num_computed_tokens == token_budget

    # Simulate model execution
    model_output1 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[np.array([])],  # Still prefilling
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_output1)
    assert requests[0].num_computed_tokens == token_budget

    # Second schedule: should process remaining tokens
    remaining = max_model_len - token_budget
    output2 = scheduler.schedule()
    assert output2.num_scheduled_tokens[requests[0].request_id] == remaining

    model_output2 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[np.array([])],  # Still prefilling
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output2, model_output2)

    # Verify all tokens were processed
    assert requests[0].num_computed_tokens == max_model_len
    assert requests[0].num_tokens == max_model_len

    # Request should complete successfully (no infinite loop)
    assert (
        requests[0].status == RequestStatus.RUNNING
        or requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    )


def test_schedule_max_model_len_no_infinite_loop():
    """Test that scheduler doesn't enter infinite loop at max capacity.

    This reproduces the bug scenario where a prompt at max_model_len
    would cause an infinite loop because the final token couldn't be scheduled.
    """
    max_model_len = 2048
    token_budget = 1024

    scheduler = create_scheduler(
        max_model_len=max_model_len,
        max_num_batched_tokens=token_budget,
    )

    # Create request with prompt exactly at max_model_len
    requests = create_requests(
        num_requests=1,
        num_tokens=max_model_len,
        max_tokens=1,
    )
    scheduler.add_request(requests[0])

    # Iteration 1: Schedule first batch
    output1 = scheduler.schedule()
    assert output1.num_scheduled_tokens[requests[0].request_id] == token_budget

    model_output1 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[np.array([])],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_output1)
    assert requests[0].num_computed_tokens == token_budget

    # Iteration 2: Schedule second batch
    output2 = scheduler.schedule()
    # Should schedule exactly token_budget tokens (not token_budget-1)
    assert output2.num_scheduled_tokens[requests[0].request_id] == token_budget

    model_output2 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[np.array([])],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output2, model_output2)
    assert requests[0].num_computed_tokens == 2 * token_budget

    # Iteration 3: Should be able to schedule the final token
    # This is where the bug would manifest - num_new_tokens would be 0
    output3 = scheduler.schedule()

    # With the fix, we should be able to schedule the remaining token
    # Without the fix, this would be 0, causing an infinite loop
    expected_remaining = max_model_len - (2 * token_budget)
    assert (
        output3.num_scheduled_tokens.get(requests[0].request_id, 0)
        == expected_remaining
    )

    # If remaining is 0, request should have finished
    if expected_remaining == 0:
        assert requests[0].num_computed_tokens == max_model_len
    else:
        model_output3 = ModelRunnerOutput(
            req_ids=[requests[0].request_id],
            req_id_to_index={requests[0].request_id: 0},
            sampled_token_ids=[np.array([])],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output3, model_output3)
        assert requests[0].num_computed_tokens == max_model_len


def test_schedule_respects_max_tokens_exactly():
    """Test that scheduler respects max_tokens without off-by-one error.

    Verifies that users receive exactly max_tokens outputs, not max_tokens-1.
    """
    max_model_len = 2048
    num_prompt_tokens = 100
    max_tokens = 50

    scheduler = create_scheduler(
        max_model_len=max_model_len,
        max_num_batched_tokens=1024,
    )

    requests = create_requests(
        num_requests=1,
        num_tokens=num_prompt_tokens,
        max_tokens=max_tokens,
    )
    scheduler.add_request(requests[0])

    # Schedule and process the prompt
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[requests[0].request_id] == num_prompt_tokens

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[np.array([1])],  # First output token
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Generate tokens until we reach max_tokens
    for i in range(1, max_tokens):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens[requests[0].request_id] == 1

        model_output = ModelRunnerOutput(
            req_ids=[requests[0].request_id],
            req_id_to_index={requests[0].request_id: 0},
            sampled_token_ids=[np.array([i + 1])],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_output)

    # Verify we generated exactly max_tokens outputs
    assert requests[0].num_output_tokens == max_tokens
    assert requests[0].num_tokens == num_prompt_tokens + max_tokens
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    "max_model_len,prompt_len,token_budget",
    [
        (2048, 2048, 512),  # Exact match
        (2048, 2047, 512),  # One token short
        (2048, 2040, 512),  # Few tokens short
    ],
)
def test_schedule_various_capacity_scenarios(max_model_len, prompt_len, token_budget):
    """Test scheduler handles various capacity scenarios correctly.

    Parametrized test to ensure the fix works across different configurations.
    """
    scheduler = create_scheduler(
        max_model_len=max_model_len,
        max_num_batched_tokens=token_budget,
    )

    requests = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=1,
    )
    scheduler.add_request(requests[0])

    total_scheduled = 0
    max_iterations = 100  # Prevent actual infinite loops in test
    iteration = 0

    while requests[0].num_computed_tokens < prompt_len and iteration < max_iterations:
        output = scheduler.schedule()

        if requests[0].request_id in output.num_scheduled_tokens:
            scheduled = output.num_scheduled_tokens[requests[0].request_id]
            total_scheduled += scheduled

            model_output = ModelRunnerOutput(
                req_ids=[requests[0].request_id],
                req_id_to_index={requests[0].request_id: 0},
                sampled_token_ids=[np.array([])],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
            scheduler.update_from_output(output, model_output)

        iteration += 1

    # Verify all tokens were scheduled
    assert total_scheduled == prompt_len, (
        f"Expected {prompt_len} tokens, but scheduled {total_scheduled}"
    )
    assert requests[0].num_computed_tokens == prompt_len
    assert iteration < max_iterations, "Scheduler entered infinite loop"
