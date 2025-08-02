# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from .utils import (create_model_runner_output, create_request,
                    create_scheduler, create_vllm_config)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int]
) -> Callable[[Request, int], tuple[int, bool]]:

    def get_num_new_matched_tokens(request: Request,
                                   _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, False

    return get_num_new_matched_tokens


@pytest.fixture
def scheduler():
    vllm_config = create_vllm_config()
    return create_scheduler(vllm_config)


@pytest.mark.parametrize(
    "num_prompt_blocks,"
    "num_external_computed_blocks,"
    "invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_non_shared_invalid_blocks(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = (num_external_computed_blocks *
                                    scheduler.block_size)

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens))
    scheduler.connector.request_finished.return_value = (False, None)

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    assert len(scheduler.running) == 3
    assert len(scheduler_output.scheduled_new_reqs) == 3
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[
            request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    model_runner_output = create_model_runner_output(
        [request1, request2, request3], use_eos=True)
    req_block_ids = scheduler_output.scheduled_new_reqs[1].block_ids[0]
    model_runner_output.invalid_block_ids = {
        req_block_ids[i]
        for i in invalid_block_idxs
    }

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == request2.request_id
    assert scheduler.running[0].num_computed_tokens == (
        min(invalid_block_idxs) * scheduler.block_size)
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3
    assert scheduler.connector.request_finished.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,"
    "num_external_computed_blocks,"
    "num_common_prefix_blocks,"
    "invalid_block_idxs",
    [
        (100, 99, 50, {0, 49}),
        (100, 99, 50, {25, 49}),
        (100, 99, 50, {49}),
    ],
)
def test_shared_invalid_blocks(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    num_common_prefix_blocks: int,
    invalid_block_idxs: set[int],
):
    assert (num_prompt_blocks >= num_external_computed_blocks >=
            num_common_prefix_blocks)

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = (num_external_computed_blocks *
                                    scheduler.block_size)
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens,
                              common_prefix_len=common_prefix_len)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens,
                              common_prefix_len=common_prefix_len)
    scheduler.add_request(request=request2)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens))

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    assert len(scheduler_output.scheduled_new_reqs) == 2
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[
            request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2

    # Simulate a failure in loading some of the shared blocks.
    model_runner_output = create_model_runner_output([request1, request2],
                                                     use_eos=True)
    req1_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    model_runner_output.invalid_block_ids = {
        req1_block_ids[i]
        for i in invalid_block_idxs
    }

    scheduler.update_from_output(scheduler_output, model_runner_output)

    # req_id -> num_computed_tokens
    # all the common prefix blocks will be computed by request1
    expected_computed_tokens = {
        request1.request_id: min(invalid_block_idxs) * scheduler.block_size,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    for request in scheduler.running:
        assert request.num_computed_tokens == expected_computed_tokens[
            request.request_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2
