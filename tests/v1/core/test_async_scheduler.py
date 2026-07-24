# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import deque
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar
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
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
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
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = set()
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
        return None, None

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


def test_no_placeholder_underflow_on_discarded_spec_frame():
    num_spec = 5
    frame_size = 1 + num_spec
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

    # A resumed frame is pending while a stale spec frame returns.
    req.num_output_placeholders = 1
    req.async_tokens_to_discard = frame_size
    computed_before = req.num_computed_tokens

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: frame_size},
        total_num_scheduled_tokens=frame_size,
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

    assert req.num_output_placeholders == 1
    assert req.num_computed_tokens == computed_before
    assert req.async_tokens_to_discard == 0
    assert req.status == RequestStatus.RUNNING


@pytest.mark.parametrize("num_spec", [0, 1, 3, 5])
def test_preempt_drain_matches_inflight_frames(num_spec: int):
    """A stale preempted frame must consume its full placeholder budget."""
    kwargs: dict = dict(async_scheduling=True, max_num_seqs=4)
    if num_spec > 0:
        kwargs["num_speculative_tokens"] = num_spec
        kwargs["speculative_method"] = "ngram_gpu"

    scheduler = create_scheduler(**kwargs)
    (req,) = create_requests(num_requests=1, num_tokens=8, max_tokens=32)
    scheduler.add_request(req)

    so_prefill = scheduler.schedule()
    assert so_prefill.num_scheduled_tokens[req.request_id] == req.num_prompt_tokens
    scheduler.update_from_output(so_prefill, _make_model_runner_output(so_prefill))
    initial_output_len = len(req.output_token_ids)
    assert initial_output_len == 1
    assert req.num_output_placeholders == 0

    if num_spec > 0:
        scheduler.update_draft_token_ids(
            DraftTokenIds([req.request_id], [list(range(100, 100 + num_spec))])
        )

    so_decode = scheduler.schedule()
    expected_tokens = 1 + num_spec
    assert so_decode.num_scheduled_tokens[req.request_id] == expected_tokens
    assert req.num_output_placeholders == expected_tokens

    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.status == RequestStatus.PREEMPTED
    assert req.num_output_placeholders == 0
    assert req.async_tokens_to_discard == expected_tokens

    mro_stale = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999] * expected_tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(so_decode, mro_stale)
    assert len(req.output_token_ids) == initial_output_len

    assert req.async_tokens_to_discard == 0

    so_re_prefill = scheduler.schedule()
    assert so_re_prefill.num_scheduled_tokens.get(req.request_id, 0) > 0
    scheduler.update_from_output(
        so_re_prefill, _make_model_runner_output(so_re_prefill)
    )
    assert len(req.output_token_ids) == initial_output_len + 1

    if num_spec > 0:
        scheduler.update_draft_token_ids(
            DraftTokenIds([req.request_id], [list(range(200, 200 + num_spec))])
        )
    so_decode2 = scheduler.schedule()
    assert so_decode2.num_scheduled_tokens.get(req.request_id, 0) > 0
    len_before = len(req.output_token_ids)
    scheduler.update_from_output(so_decode2, _make_model_runner_output(so_decode2))
    assert len(req.output_token_ids) > len_before


def _make_spec_scheduler_output(
    req_id: str, num_sampled: int, num_scheduled_spec: int
) -> SchedulerOutput:
    frame_size = num_sampled + num_scheduled_spec
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: frame_size},
        total_num_scheduled_tokens=frame_size,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens=(
            {req_id: [10] * num_scheduled_spec} if num_scheduled_spec else {}
        ),
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_preempt_with_zero_placeholders_is_noop():
    """Preempting with no in-flight frame must not add a discard obligation."""
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
    )
    (req,) = create_requests(num_requests=1, num_tokens=8, max_tokens=32)
    scheduler.add_request(req)

    so_prefill = scheduler.schedule()
    scheduler.update_from_output(so_prefill, _make_model_runner_output(so_prefill))
    assert req.num_output_placeholders == 0
    assert req.async_tokens_to_discard == 0

    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())

    assert req.async_tokens_to_discard == 0
    assert req.num_output_placeholders == 0


def test_repreempt_accumulates_async_tokens_to_discard():
    """Re-preemption must preserve the prior stale-frame discard obligation."""
    num_spec = 3
    frame_size = 1 + num_spec
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )
    (req,) = create_requests(num_requests=1, num_tokens=8, max_tokens=64)
    scheduler.add_request(req)

    so_prefill = scheduler.schedule()
    scheduler.update_from_output(so_prefill, _make_model_runner_output(so_prefill))
    scheduler.update_draft_token_ids(
        DraftTokenIds([req.request_id], [list(range(100, 100 + num_spec))])
    )
    scheduler.schedule()
    assert req.num_output_placeholders == frame_size

    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.async_tokens_to_discard == frame_size
    assert req.num_output_placeholders == 0

    so_resume = scheduler.schedule()
    assert req.request_id in so_resume.num_scheduled_tokens
    assert req.status == RequestStatus.RUNNING
    leftover = req.async_tokens_to_discard
    assert leftover == frame_size
    new_placeholders = req.num_output_placeholders

    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.async_tokens_to_discard == leftover + new_placeholders, (
        f"expected {leftover + new_placeholders}, got {req.async_tokens_to_discard}"
    )
    assert req.num_output_placeholders == 0

    req.status = RequestStatus.RUNNING
    if req not in scheduler.running:
        scheduler.running.append(req)
    leftover = req.async_tokens_to_discard
    req.num_output_placeholders = frame_size
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.async_tokens_to_discard == leftover + frame_size
    assert req.num_output_placeholders == 0

    before = req.async_tokens_to_discard
    assert before > 0
    req.status = RequestStatus.RUNNING
    if req not in scheduler.running:
        scheduler.running.append(req)
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.async_tokens_to_discard == before


@pytest.mark.parametrize("num_spec", [0, 1, 3, 5])
def test_preempt_drain_ignores_actual_token_count(num_spec: int):
    """Discard accounting must use the scheduled frame size, not its output."""
    kwargs: dict = dict(async_scheduling=True, max_num_seqs=4)
    if num_spec > 0:
        kwargs["num_speculative_tokens"] = num_spec
        kwargs["speculative_method"] = "ngram_gpu"

    scheduler = create_scheduler(**kwargs)
    (req,) = create_requests(num_requests=1, num_tokens=8, max_tokens=32)
    scheduler.add_request(req)

    so_prefill = scheduler.schedule()
    scheduler.update_from_output(so_prefill, _make_model_runner_output(so_prefill))
    if num_spec > 0:
        scheduler.update_draft_token_ids(
            DraftTokenIds([req.request_id], [list(range(100, 100 + num_spec))])
        )

    so_decode = scheduler.schedule()
    frame_size = 1 + num_spec
    assert req.num_output_placeholders == frame_size

    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert req.async_tokens_to_discard == frame_size

    mro_partial = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(so_decode, mro_partial)

    assert req.async_tokens_to_discard == 0


@pytest.mark.parametrize("num_spec", [0, 3])
def test_reset_prefix_cache_drains_spec_frames(num_spec: int):
    """Prefix-cache reset must drain stale frames through preemption."""
    kwargs: dict = dict(async_scheduling=True, max_num_seqs=4)
    if num_spec > 0:
        kwargs["num_speculative_tokens"] = num_spec
        kwargs["speculative_method"] = "ngram_gpu"

    scheduler = create_scheduler(**kwargs)
    (req,) = create_requests(num_requests=1, num_tokens=8, max_tokens=32)
    scheduler.add_request(req)

    so_prefill = scheduler.schedule()
    scheduler.update_from_output(so_prefill, _make_model_runner_output(so_prefill))
    if num_spec > 0:
        scheduler.update_draft_token_ids(
            DraftTokenIds([req.request_id], [list(range(100, 100 + num_spec))])
        )

    so_decode = scheduler.schedule()
    frame_size = 1 + num_spec
    assert req.num_output_placeholders == frame_size

    scheduler.reset_prefix_cache(reset_running_requests=True)
    assert req.status == RequestStatus.PREEMPTED
    assert req.async_tokens_to_discard == frame_size
    assert req.num_output_placeholders == 0

    mro_stale = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999] * frame_size],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(so_decode, mro_stale)
    assert req.async_tokens_to_discard == 0


@pytest.mark.parametrize(
    "frame_sizes",
    [
        [1, 1],
        [4, 4],
        [4, 1],
        [1, 4],
        [4, 4, 4],
    ],
    ids=[
        "two_non_spec",
        "two_spec",
        "spec_then_non_spec",
        "non_spec_then_spec",
        "three_spec",
    ],
)
def test_multi_inflight_frames_drain_by_own_frame_size(frame_sizes: list[int]):
    """Each stale frame must consume its own scheduled placeholder budget."""
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
    )
    (req,) = create_requests(num_requests=1, max_tokens=20)
    req.num_computed_tokens = req.num_tokens
    scheduler.requests[req.request_id] = req
    scheduler.running.append(req)
    req.status = RequestStatus.RUNNING

    req.num_output_placeholders = 0
    req.async_tokens_to_discard = sum(frame_sizes)

    running_total = req.async_tokens_to_discard
    for frame_size in frame_sizes:
        num_scheduled_spec = frame_size - 1
        scheduler_output = _make_spec_scheduler_output(
            req.request_id, num_sampled=1, num_scheduled_spec=num_scheduled_spec
        )
        model_runner_output = ModelRunnerOutput(
            req_ids=[req.request_id],
            req_id_to_index={req.request_id: 0},
            sampled_token_ids=[[999] * frame_size],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(scheduler_output, model_runner_output)
        running_total -= frame_size
        assert req.async_tokens_to_discard == running_total

    assert req.async_tokens_to_discard == 0
    assert req.num_output_placeholders == 0
