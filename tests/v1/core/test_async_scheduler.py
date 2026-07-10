# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
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
    assert sched_output.num_scheduled_tokens[req0.request_id] == num_prompt_tokens
    assert (
        sched_output.num_scheduled_tokens[req1.request_id]
        == num_prompt_tokens % BLOCK_SIZE
    )

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        added_req = None
        if requests:
            added_req = requests.pop(0)
            scheduler.add_request(added_req)
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
            if added_req:
                assert (
                    sched_output.num_scheduled_tokens[added_req.request_id]
                    == num_prompt_tokens % BLOCK_SIZE
                )

    assert scheduler.get_num_unfinished_requests() == 0


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
        req.update_block_hashes()

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert sched_output.num_scheduled_tokens[req.request_id] == (
            req.num_prompt_tokens % BLOCK_SIZE
        )


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(AsyncScheduler)
    request = create_requests(num_requests=1, num_tokens=1)[0]
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock()
    request.structured_output_request.grammar.accept_tokens.return_value = False
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.structured_output_manager.trim_reasoning_for_advance.side_effect = (
        lambda request, new_token_ids: new_token_ids
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(output, model_runner_output)

    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running


def test_no_placeholder_underflow_on_stale_spec_output():
    """A request's stale in-flight spec output (from a step scheduled before a
    preemption rolled it back, now resumed) must NOT apply its pre-reset
    rejection count to the resumed request's freshly re-added placeholder count
    -- that underflows ``num_output_placeholders`` below zero. Its token is
    still delivered.
    """
    num_spec = 5
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )
    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.num_computed_tokens = req.num_tokens
    scheduler.requests[req.request_id] = req
    req.status = RequestStatus.PREEMPTED

    # A small placeholder count as if the request has just resumed, plus one
    # stale in-flight output outstanding (1 sampled + num_spec draft tokens).
    req.num_output_placeholders = 1
    req.num_stale_output_tokens = num_spec + 1
    computed_before = req.num_computed_tokens
    outputs_before = len(req.output_token_ids)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: num_spec + 1},
        total_num_scheduled_tokens=num_spec + 1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={req.request_id: [10] * num_spec},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Stale output's counter mutations are skipped (no underflow/corruption)...
    assert req.num_output_placeholders == 1
    assert req.num_computed_tokens == computed_before
    assert req.num_stale_output_tokens == 0
    # ...but its token is still delivered (lossless).
    assert len(req.output_token_ids) == outputs_before + 1


def test_preempt_marks_all_inflight_async_output_stale():
    """A KV-pressure preemption must account for *every* piece of the request's
    in-flight output, not just one. ``_preempt_request`` records the request's
    outstanding ``num_in_flight_tokens`` as stale; each stale return then
    delivers its token (lossless) but is drained by its own scheduled token
    count, so the reset ``num_output_placeholders`` never underflows and the
    resumed request's counters are never corrupted by a stale rejection count.

    Regression: the first attempt discarded a single return, assuming at most
    one is in flight -- but async depth / PP / spec can leave several, and the
    leftover stale returns underflowed the placeholder count.
    """
    num_spec = 4
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )
    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.num_computed_tokens = req.num_tokens
    scheduler.requests[req.request_id] = req
    scheduler.running.append(req)
    req.status = RequestStatus.RUNNING
    # The request has two in-flight spec-decode outputs (async depth / PP): each
    # reserves num_sampled_tokens_per_step (1) + num_spec draft tokens.
    scheduled_tokens = scheduler.num_sampled_tokens_per_step + num_spec
    req.num_output_placeholders = 2 * scheduled_tokens
    req.num_in_flight_tokens = 2 * scheduled_tokens

    scheduler.running.remove(req)
    scheduler._preempt_request(req, timestamp=0.0)

    assert req.status == RequestStatus.PREEMPTED
    # The whole in-flight token count is marked stale -- not one output.
    assert req.num_stale_output_tokens == 2 * scheduled_tokens
    assert req.num_output_placeholders == 0
    outputs_before = len(req.output_token_ids)

    def _return_stale_output():
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req.request_id: scheduled_tokens},
            total_num_scheduled_tokens=scheduled_tokens,
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={req.request_id: [10] * num_spec},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        model_runner_output = ModelRunnerOutput(
            req_ids=[req.request_id],
            req_id_to_index={req.request_id: 0},
            sampled_token_ids=[[999]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Both stale outputs return: each delivers its token, drains the stale
    # count, and leaves the reset placeholder count untouched (no underflow).
    _return_stale_output()
    assert req.num_stale_output_tokens == scheduled_tokens
    _return_stale_output()
    assert req.num_stale_output_tokens == 0
    assert req.num_output_placeholders == 0
    assert req.status == RequestStatus.PREEMPTED
    # Each stale output delivered its sampled token (lossless).
    assert len(req.output_token_ids) == outputs_before + 2


def test_reset_preempt_drops_inflight_async_output():
    """reset_prefix_cache preempts and resumes in the same step, so a kept token
    would arrive out of order. ``_preempt_request(drop_stale_output=True)`` marks
    the in-flight output stale-and-dropped: it drains the stale count without
    delivering a token or touching the reset counters.
    """
    num_spec = 4
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )
    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.num_computed_tokens = req.num_tokens
    scheduler.requests[req.request_id] = req
    scheduler.running.append(req)
    req.status = RequestStatus.RUNNING
    scheduled_tokens = scheduler.num_sampled_tokens_per_step + num_spec
    req.num_output_placeholders = scheduled_tokens
    req.num_in_flight_tokens = scheduled_tokens

    scheduler.running.remove(req)
    scheduler._preempt_request(req, timestamp=0.0, drop_stale_output=True)

    assert req.drop_stale_output is True
    assert req.num_stale_output_tokens == scheduled_tokens
    outputs_before = len(req.output_token_ids)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: scheduled_tokens},
        total_num_scheduled_tokens=scheduled_tokens,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={req.request_id: [10] * num_spec},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # The stale output is dropped: no token delivered, count drained, no underflow.
    assert req.num_stale_output_tokens == 0
    assert req.num_output_placeholders == 0
    assert len(req.output_token_ids) == outputs_before
    assert req.status == RequestStatus.PREEMPTED
