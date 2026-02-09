# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_stop_by_max_tokens(max_tokens: int):
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=2, max_tokens=max_tokens)
    req0, req1 = requests

    expected_total_num_scheduled_tokens = 0
    sched_outputs: deque[SchedulerOutput] = deque()
    scheduler.add_request(req0)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req0.num_prompt_tokens + max_tokens - 1

    scheduler.add_request(req1)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req1.num_prompt_tokens + max_tokens - 1

    total_num_scheduled_tokens = 0
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        total_num_scheduled_tokens += sched_output.total_num_scheduled_tokens
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req0.num_output_tokens == max_tokens
    assert req1.num_output_tokens == max_tokens
    # Ensure we aren't scheduling more tokens than necessary.
    assert total_num_scheduled_tokens == expected_total_num_scheduled_tokens


def test_abort():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_preempt():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_prefix_caching_for_prefill_dedup():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=3,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )
    requests_copy = requests.copy()

    # Two requests with the same prompt.
    req0 = requests.pop(0)
    req1 = requests.pop(0)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)
    # Make sure prefix caching de-duplicates the prompts in the same step,
    # so all the blocks except the last are shared between the two requests.
    assert len(sched_output.num_scheduled_tokens) == 2
    num_blocks = num_prompt_tokens // BLOCK_SIZE
    assert req0.num_cached_tokens == 0
    assert req1.num_cached_tokens >= num_blocks * BLOCK_SIZE

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        if requests:
            scheduler.add_request(requests.pop(0))
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    # Other requests scheduled after the two requests should also get
    # prefix cache hit.
    assert scheduler.get_num_unfinished_requests() == 0
    for req in requests_copy[1:]:
        assert req.num_cached_tokens >= num_blocks * BLOCK_SIZE


def test_prefix_caching_for_multi_turn():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    num_output_tokens = 200
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )

    for req in requests:
        scheduler.add_request(req)
    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    # Process the requests.
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
    assert scheduler.get_num_unfinished_requests() == 0

    # Create next-turn requests whose prompts are the full output of the
    # previous turn.
    next_turn_requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens + num_output_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )
    for i, req in enumerate(next_turn_requests):
        req.prompt_token_ids = requests[i].prompt_token_ids + list(
            requests[i].output_token_ids
        )
        req._all_token_ids = req.prompt_token_ids.copy()
        req.all_token_ids = ConstantList(req._all_token_ids)
        req.block_hashes = []
        req.block_hashes = req._block_hasher(req)

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_outputs.append(scheduler.schedule())

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert req.num_cached_tokens == req.num_prompt_tokens // BLOCK_SIZE * BLOCK_SIZE
