# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
from unittest.mock import Mock

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ECTransferConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test


def test_add_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


def test_finish_request():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule(enable_prefix_caching: bool, prompt_logprobs: int | None):
    """Test scheduling.
    Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
    """
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10, prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # Verify all requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request


def test_schedule_multimodal_requests():
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_positions = [[PlaceholderRange(offset=i, length=100)] for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)
    assert len(output.scheduled_encoder_inputs) == 10
    for req_id, encoder_input in output.scheduled_encoder_inputs.items():
        assert len(encoder_input) == 1


def test_async_scheduling_pp_allows_rescheduling_with_output_placeholders():
    """Async scheduling + PP: allow multi-step in-flight scheduling per request"""
    scheduler = create_scheduler(async_scheduling=True, pipeline_parallel_size=2)
    (req,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(req)

    _ = scheduler.schedule()
    assert req.num_output_placeholders > 0

    # before any update_from_output, we still expect the request can be
    # scheduled again (multi-step in-flight).
    output = scheduler.schedule()
    assert req.request_id in output.num_scheduled_tokens


def test_schedule_partial_requests():
    """Test scheduling behavior with partial requests.

    This test verifies that:
    1. The scheduler can handle multiple partial requests in a single step when
       constrained by encoder budget.
    2. A request in RUNNING state may be unscheduled in subsequent steps if
       there is insufficient encoder budget.
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
    )
    mm_positions = [[PlaceholderRange(offset=100, length=600)] for _ in range(3)]
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    assert scheduler.max_num_encoder_input_tokens == 1024
    # The first request is scheduled fully.
    assert output.num_scheduled_tokens[requests[0].request_id] == 800
    # The second request is scheduled partially.
    # The <img> tokens are not scheduled because of the encoder budget.
    assert output.num_scheduled_tokens[requests[1].request_id] == 100
    # The third request is also scheduled partially.
    # The <img> tokens are not scheduled because of the encoder budget.
    assert output.num_scheduled_tokens[requests[2].request_id] == 100
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], [], []],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    # Only the first and second requests are scheduled.
    # The third request is in the RUNNING state but not scheduled in this step
    # because of the encoder budget.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 2
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 700
    assert requests[2].request_id not in output.num_scheduled_tokens


def test_no_mm_input_chunking():
    # Disable multimodal input chunking.
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        max_model_len=2048,
    )
    mm_positions = [[PlaceholderRange(offset=400, length=800)]]
    requests = create_requests(
        num_requests=1, num_tokens=1200, mm_positions=mm_positions
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # We want to only see the 400 text tokens at the start scheduled
    assert output.num_scheduled_tokens[requests[0].request_id] == 400

    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 800

    # Test that we fail if we disable chunked mm input and use too small
    # of a max_num_batched_tokens for the mm input.
    with pytest.raises(ValueError):
        _ = create_scheduler(
            model="llava-hf/llava-1.5-7b-hf",
            max_num_batched_tokens=100,
            disable_chunked_mm_input=True,
        )


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_schedule_concurrent_partial_requests(enable_prefix_caching: bool):
    """Test scheduling behavior with concurrent partial requests.

    This test verifies that: there are multiple long prefill requests in the
    RUNNING state, and we can schedule them together.

    """
    scheduler = create_scheduler(
        model="facebook/opt-125m",
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[0].request_id] == 400
    # The second request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[1].request_id] == 400
    # The third request is also scheduled partially - 1024 - 400 - 400 = 224.
    assert output.num_scheduled_tokens[requests[2].request_id] == 224
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step. All three requests are running.
    # Processed the remaining prefills of the first and second requests.
    output1 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output1.scheduled_new_reqs) == 0
    assert output1.scheduled_cached_reqs.num_reqs == 3
    assert len(output1.finished_req_ids) == 0
    assert output1.num_scheduled_tokens[requests[0].request_id] == 400
    assert output1.num_scheduled_tokens[requests[1].request_id] == 400
    assert output1.num_scheduled_tokens[requests[2].request_id] == 224

    # Schedule the third step. All three requests are running.
    # First and second requests are in the decode stage.
    # All the remaining tokens in the third request are processed.
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0], [0]] + [[] for _ in range(len(requests) - 2)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_runner_output)
    output2 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output2.scheduled_new_reqs) == 0
    assert output2.scheduled_cached_reqs.num_reqs == 3
    assert len(output2.finished_req_ids) == 0
    assert output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert output2.num_scheduled_tokens[requests[1].request_id] == 1
    assert output2.num_scheduled_tokens[requests[2].request_id] == 800 - 224 - 224


def test_stop_via_update_from_output():
    """Test stopping behavior through update_from_output"""
    scheduler = create_scheduler(num_speculative_tokens=1)

    # Test case 1: Stop on EOS token
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 1, requests[1].request_id: 2},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [],
            requests[1].request_id: [10],
        },
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[
            [EOS_TOKEN_ID],
            [10, 11],
        ],  # First request hits EOS, second continues
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped, second continues
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID]
    assert list(requests[1].output_token_ids) == [10, 11]

    # Test case 2: Stop on custom stop token
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=10, stop_token_ids=[42, 43])
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 2},
        total_num_scheduled_tokens=5,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [10, 42],
            requests[1].request_id: [13],
        },
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 42, 12], [13, 14]],  # First request hits stop token
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped on custom token
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].stop_reason == 42
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 42]
    assert list(requests[1].output_token_ids) == [13, 14]

    # Test case 3: Stop on max tokens
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=2)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 1},
        total_num_scheduled_tokens=4,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [10, 11],
            requests[1].request_id: [],
        },
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 11, 12], [13]],  # First request exceeds max_tokens
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped due to length
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 11]  # Truncated to max_tokens
    assert list(requests[1].output_token_ids) == [13]

    # Test case 4: Ignore EOS flag
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=1, max_tokens=10)
    requests[0].sampling_params.ignore_eos = True
    requests[0].num_computed_tokens = requests[0].num_tokens
    scheduler.requests[requests[0].request_id] = requests[0]
    scheduler.running.append(requests[0])

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={requests[0].request_id: [EOS_TOKEN_ID, 10]},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify request continues past EOS
    assert len(scheduler.running) == 1
    assert not requests[0].is_finished()
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID, 10, 11]


def test_check_stop_min_tokens():
    """Test that requests don't stop when min_tokens requirement isn't met."""
    from vllm.v1.core.sched.utils import check_stop

    # Test case 1: num_output_tokens < min_tokens
    # Should return False (don't stop)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
    )
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
    )
    # Simulate having generated 3 output tokens (less than min_tokens=5)
    request.append_output_token_ids([10, 11, EOS_TOKEN_ID])  # EOS token present

    result = check_stop(request, max_model_len=100)
    assert result is False, "Should not stop when num_output_tokens<min_tokens"

    # Test case 2: num_output_tokens >= min_tokens
    # Should follow normal stopping logic (stop on EOS)
    request.append_output_token_ids(
        [
            10,
            11,
            12,
            13,
            14,
            EOS_TOKEN_ID,
        ]
    )  # 6 tokens > min_tokens

    result = check_stop(request, max_model_len=100)
    assert result is True, "Should stop on EOS when min_tokens met"
    assert request.status == RequestStatus.FINISHED_STOPPED

    # Test case 3: min_tokens = 0, should follow normal stopping logic
    sampling_params_no_min = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=0,
    )
    request_no_min = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_no_min,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
    )
    request_no_min.append_output_token_ids([10, EOS_TOKEN_ID])

    result = check_stop(request_no_min, max_model_len=100)
    assert result is True, "Should stop on EOS when min_tokens=0"
    assert request_no_min.status == RequestStatus.FINISHED_STOPPED

    # Test case 4: min_tokens > 0 with stop token (not EOS)
    sampling_params_stop = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
        stop_token_ids=[42],
    )
    request_stop = Request(
        request_id="2",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_stop,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
    )
    # Only 3 output tokens, less than min_tokens=5, but has stop token
    request_stop.append_output_token_ids([10, 11, 42])
    result = check_stop(request_stop, max_model_len=100)
    assert result is False, "Should not stop when num_output_tokens<min_tokens"

    # Test case 5: min_tokens met, should stop on stop token
    request_stop.append_output_token_ids(
        [10, 11, 12, 13, 14, 42]
    )  # 6 tokens >= min_tokens=5

    result = check_stop(request_stop, max_model_len=100)
    assert result is True, "Should stop on stop token when min_tokens met"
    assert request_stop.status == RequestStatus.FINISHED_STOPPED
    assert request_stop.stop_reason == 42


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule_concurrent_batches(
    enable_prefix_caching: bool, prompt_logprobs: int | None
):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=512,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[requests[0].request_id] == 512

    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[requests[1].request_id] == 512

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Schedule the next step.
    # The first request can be scheduled again while the second
    # request is still running.
    scheduler_output2 = scheduler.schedule()
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)


@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
def test_schedule_order(enable_chunked_prefill: bool):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=3,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    # long requests
    requests = create_requests(num_requests=2, num_tokens=800, req_ids=["1", "2"])
    # short requests
    requests += create_requests(num_requests=2, num_tokens=10, req_ids=["3", "4"])

    for request in requests:
        scheduler.add_request(request)

    scheduler_output1 = scheduler.schedule()

    if enable_chunked_prefill:
        # When enable chunked prefill, long requests will be chunked.
        assert len(scheduler_output1.scheduled_new_reqs) == 2
    else:
        # When disable chunked prefill, should not skip the long requests,
        # and scheduling subsequent short requests in advance,
        # even though there is still token budgets remaining.
        assert len(scheduler_output1.scheduled_new_reqs) == 1


def test_preempt_during_execution():
    # NOTE(woosuk): The actual number of available blocks is 10 instead of 11
    # because block 0 is reserved as the null block.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.num_scheduled_tokens) == 1
    assert len(scheduler_output0.scheduled_new_reqs[0].block_ids[0]) == 5

    # Schedule the second request while the first request is still running.
    # This scenario can occur in certain cases, when max_concurrent_batches > 1
    # (e.g., when pipeline parallelism is used).
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.num_scheduled_tokens) == 1
    assert len(scheduler_output1.scheduled_new_reqs[0].block_ids[0]) == 5

    # Get the output of the first request.
    model_runner_output0 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output0)

    # Schedule the first request again. This will cause the preemption
    # of the second request because the KV cache is full.
    _ = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert scheduler.running[0] == requests[0]
    assert requests[1].status == RequestStatus.PREEMPTED

    model_runner_output1 = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output1)

    # The second request (that is preempted) should be updated with the
    # sampled token id.
    assert len(requests[1].output_token_ids) == 1
    assert requests[1].output_token_ids[0] == 42


def test_scheduler_reset_prefix_cache():
    scheduler = create_scheduler(enable_prefix_caching=True)
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    # Initial scheduling, requests should be at the running state now
    _ = scheduler.schedule()

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request

    # Reset prefix cache should fail since there are still running requests
    # and they are taking KV cache
    assert not scheduler.reset_prefix_cache()

    # Reset prefix cache with reset_running_requests=True. All running requests
    # Should be pushed back to the waiting queue and kv cache should be freed
    assert scheduler.reset_prefix_cache(reset_running_requests=True)

    # Verify requests moved from running to waiting
    assert len(scheduler.waiting) == len(requests)
    assert len(scheduler.running) == 0

    for i, request in enumerate(requests):
        assert scheduler.waiting[i] == request


# Note - these test cases mirror some of those in test_rejection_sampler.py
@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2, 3]], [[1, 2, 3, 4]], (1, 3, 3, [1, 1, 1])),  # perfect match
        ([[1, 2, 3]], [[1, 5]], (1, 3, 1, [1, 0, 0])),  # early mismatch
        ([[1, 2], [3]], [[1, 2, 5], [3, 4]], (2, 3, 3, [2, 1])),  # multiple sequences
        ([[1]], [[1, 2]], (1, 1, 1, [1])),  # single token sequence
        ([[]], [[5]], (0, 0, 0, [0])),  # empty sequence
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 7], [4, 8]],
            (2, 6, 3, [2, 1, 0]),
        ),  # multiple mismatches
    ],
)
def test_schedule_spec_decoding_stats(spec_tokens, output_tokens, expected):
    """Test scheduling behavior with speculative decoding.

    This test verifies that:
    1. Speculated tokens get scheduled correctly
    2. Spec decoding stats properly count number of draft and accepted tokens
    """
    num_spec_tokens = max(1, max(len(t) for t in spec_tokens))
    scheduler = create_scheduler(num_speculative_tokens=num_spec_tokens)
    requests = create_requests(num_requests=len(spec_tokens), num_tokens=1)
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Schedule a decode, which will also draft speculative tokens
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.total_num_scheduled_tokens == len(requests)
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1
        assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    draft_token_ids = DraftTokenIds(req_ids, spec_tokens)
    scheduler.update_draft_token_ids(draft_token_ids)

    for i in range(len(requests)):
        running_req = scheduler.running[i]
        # The prompt token
        assert running_req.num_computed_tokens == 1
        # The prompt token and the sampled token
        assert running_req.num_tokens == 2
        # The prompt token, the sampled token, and the speculated tokens
        assert running_req.num_tokens_with_spec == 2 + len(spec_tokens[i])

    # No draft or accepted tokens counted yet
    assert not engine_core_outputs or (
        engine_core_outputs[0].scheduler_stats.spec_decoding_stats is None
    )

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    # The sampled token and speculated tokens
    assert output.total_num_scheduled_tokens == len(requests) + sum(
        len(ids) for ids in spec_tokens
    )
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1 + len(spec_tokens[i])
        if spec_tokens[i]:
            assert len(output.scheduled_spec_decode_tokens[req_id]) == len(
                spec_tokens[i]
            )
        else:
            assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=output_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    scheduler_stats = (
        engine_core_outputs[0].scheduler_stats if engine_core_outputs else None
    )
    if expected[0] == 0:
        assert scheduler_stats is not None
        assert scheduler_stats.spec_decoding_stats is None
    else:
        assert scheduler_stats is not None
        assert scheduler_stats.spec_decoding_stats is not None
        stats = scheduler_stats.spec_decoding_stats
        assert stats.num_drafts == expected[0]
        assert stats.num_draft_tokens == expected[1]
        assert stats.num_accepted_tokens == expected[2]
        assert stats.num_accepted_tokens_per_pos == expected[3]


def test_spec_decoding_stats_empty_output():
    """Test that spec decoding stats handle empty output tokens gracefully.

    This is a regression test for a bug where empty sampled_token_ids
    would cause num_accepted = len([]) - 1 = -1, leading to a
    ValueError when incrementing a Prometheus counter with a negative value.
    """
    num_spec_tokens = 3
    scheduler = create_scheduler(num_speculative_tokens=num_spec_tokens)
    requests = create_requests(num_requests=1, num_tokens=1)
    request = requests[0]
    req_id = request.request_id

    scheduler.add_request(request)

    # Initial schedule (prefill)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1

    # Complete the prefill with a sampled token
    model_runner_output = ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Add draft tokens for speculation
    draft_token_ids = DraftTokenIds([req_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert req_id in output.scheduled_spec_decode_tokens
    assert len(output.scheduled_spec_decode_tokens[req_id]) == 3

    # Simulate empty output tokens (e.g., due to request abortion or error)
    # This would previously cause num_accepted = -1 and crash
    model_runner_output = ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[]],  # Empty output tokens
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # This should not raise an error
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    # Spec decoding stats should be None since no tokens were generated
    scheduler_stats = (
        engine_core_outputs[0].scheduler_stats if engine_core_outputs else None
    )
    assert scheduler_stats is None or scheduler_stats.spec_decoding_stats is None


def _assert_right_scheduler_output(
    output: SchedulerOutput,
    num_requests: int,
    expected_num_scheduled_tokens: int,
):
    """Check if SchedulerOutput is correct after remote KV cache hit."""

    # We should inject the kv_connector_metadata.
    assert len(output.kv_connector_metadata.requests) == num_requests

    # Only num_tokens - matched_num_new_tokens should be scheduled.
    for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
        assert num_scheduled_tokens == expected_num_scheduled_tokens


def _assert_right_kv_cache_manager(
    scheduler: Scheduler,
    requests: list[Request],
    num_tokens: int,
    block_size: int,
    num_requests: int,
    num_total_blocks: int,
):
    """Check whether KVCacheManager is correct after allocate."""

    # Make sure the request stats are right.
    EXPECTED_TOTAL_BLOCKS = num_tokens // block_size
    for req in requests:
        blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks[req.request_id]
        hashes = req.block_hashes
        assert (
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block[req.request_id]
            == EXPECTED_TOTAL_BLOCKS
        )
        assert len(blocks) == EXPECTED_TOTAL_BLOCKS
        assert len(hashes) == EXPECTED_TOTAL_BLOCKS

    # Make sure we actually touched all the blocks.
    BLOCKS_PER_REQ = num_tokens / block_size
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        == num_total_blocks - num_requests * BLOCKS_PER_REQ
    )


def _step_until_done(
    scheduler: Scheduler,
    output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
):
    """Loop over schedule(), update_from_output() until finished."""

    all_finished = False
    _ = scheduler.update_from_output(output, model_runner_output)
    while not all_finished:
        # Schedule + a few iterations until stopping.
        output = scheduler.schedule()
        assert len(scheduler.running)
        for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
            # We should be in the decode phase now.
            assert num_scheduled_tokens == 1
        if scheduler.connector is not None:
            assert len(output.kv_connector_metadata.requests) == 0
        if scheduler.ec_connector is not None:
            assert len(output.ec_connector_metadata.mm_datas) == 0
        ecos = scheduler.update_from_output(output, model_runner_output)[0]
        all_done = True
        for eco in ecos.outputs:
            if eco.finish_reason is None:
                all_done = False
        all_finished = all_done


def _step_until_kv_transfer_finished(scheduler: Scheduler, req_ids: list[str]):
    """Cycle requests through a KV transfer cyle."""

    # Requests should first transition to WAITING_FOR_REMOTE_KVS
    output = scheduler.schedule()
    assert len(scheduler.waiting) == len(req_ids)
    assert len(scheduler.running) == 0
    assert len(output.scheduled_new_reqs) == 0
    for req in scheduler.requests.values():
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # No model execution yet
    EMPTY_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, EMPTY_OUTPUT)

    # Simulate KV transfer completion using KVConnectorOutput.finished_recving
    output = scheduler.schedule()
    assert len(scheduler.waiting) == len(req_ids)
    assert len(scheduler.running) == 0

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(finished_recving=req_ids),
    )
    scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    for req_id in req_ids:
        assert req_id in scheduler.finished_recving_kv_req_ids


@pytest.mark.parametrize("is_async", [False, True])
def test_kv_connector_basic(is_async: bool):
    """
    Test whether Scheduler with KVConnector schedules tokens, allocates
    memory, and cleans up requests as expected under normal operation.
    """

    # Setup Scheduler.
    BLOCK_SIZE = 16
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
    )
    NUM_TOTAL_BLOCKS = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    ######################################################
    # FIRST SET OF REQUESTS - External Hit Only
    NUM_REQUESTS = 2
    NUM_TOKENS = NUM_MATCHED_NEW_TOKENS * 2
    MAX_TOKENS = 3
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Ensure ScheduleOutput is correct.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens should be scheduled.
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(
        scheduler, requests, NUM_TOKENS, BLOCK_SIZE, NUM_REQUESTS, NUM_TOTAL_BLOCKS
    )

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_TOTAL_BLOCKS
    )

    ######################################################
    # SECOND SET OF REQUESTS - Local And External Hit
    NUM_TOKENS_PREFIX = NUM_TOKENS
    # We will get a local prefix cache hit for the first
    # NUM_TOKENS_PREFIX tokens since they are used above.
    NUM_TOKENS = NUM_TOKENS_PREFIX * 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # We should get a local cache hit of NUM_TOKENS_PREFIX and
    # a remote KV cache hit of NUM_MATCHED_NEW_TOKENS.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens after local + remote cache hit.
        expected_num_scheduled_tokens=(
            NUM_TOKENS - NUM_TOKENS_PREFIX - NUM_MATCHED_NEW_TOKENS
        ),
    )

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(
        scheduler, requests, NUM_TOKENS, BLOCK_SIZE, NUM_REQUESTS, NUM_TOTAL_BLOCKS
    )

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_TOTAL_BLOCKS
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_external_prefix_cache_metrics(is_async: bool):
    """
    Verify connector prefix cache metrics are updated
    correctly when the scheduler processes requests with KV connector hits.
    """

    # Setup Scheduler.
    NUM_MATCHED_NEW_TOKENS = 4
    scheduler = create_scheduler(
        enable_prefix_caching=False,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
    )

    # --- Prepare simple requests ---
    NUM_REQUESTS = 2
    NUM_TOKENS = 8
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    # --- Trigger scheduling and simulate model output ---
    output = scheduler.schedule()
    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[r.request_id for r in requests],
        req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
        sampled_token_ids=[[1000]] * NUM_REQUESTS,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Update scheduler stats
    ecos = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # --- Assertions ---
    assert ecos is not None and len(ecos) > 0
    assert ecos[0].scheduler_stats is not None

    external_stats = ecos[0].scheduler_stats.connector_prefix_cache_stats
    assert external_stats is not None

    assert external_stats.queries == NUM_TOKENS * NUM_REQUESTS
    assert external_stats.hits == NUM_MATCHED_NEW_TOKENS * NUM_REQUESTS
    assert external_stats.requests == NUM_REQUESTS
    assert external_stats.preempted_requests == 0


@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_unable_to_allocate(use_ec_connector, ec_role):
    """
    Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=False),
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
    )

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        num_requests=1,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        num_requests=1,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_handles_preemption(is_async, use_ec_connector, ec_role):
    """
    Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 2
    # NOTE: there is 1 null block, so this is 6 blocks.
    NUM_BLOCKS = 7
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
    )

    # Create two requests.
    # Both can be scheduled at first, but the second request
    # will be preempted and re-scheduled.
    NUM_REQUESTS = 2
    NUM_TOKENS = BLOCK_SIZE * 2 + 1
    MAX_TOKENS = BLOCK_SIZE * 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # All can be scheduled - 1st token.
    output = scheduler.schedule()
    if is_async:
        assert len(scheduler.waiting) == 2
        assert scheduler.running == []
        _step_until_kv_transfer_finished(scheduler, req_ids)
        output = scheduler.schedule()

    _assert_right_scheduler_output(
        output,
        # 2 remote kv cache hits.
        num_requests=2,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # All can be scheduled - 2nd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # This will generate a new block and cause a preemption - 3rd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1

    # Restarts the preempted request - generate 3rd token.
    # This will have a local and remote cache hit.
    output = scheduler.schedule()
    if is_async:
        waiting_req_ids = [req.request_id for req in scheduler.waiting]
        assert len(waiting_req_ids) == 1
        _step_until_kv_transfer_finished(scheduler, waiting_req_ids)
        output = scheduler.schedule()

    _assert_right_scheduler_output(
        output,
        # 1 remote kv_cache hit!
        num_requests=1,
        # Only 1 block was preempted and there is a single
        # remote hit. So only single new token is scheduled.
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_new_reqs == []
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_new_reqs == []
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def make_output(scheduler: Scheduler):
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={req.request_id: i for i, req in enumerate(scheduler.running)},
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        )
        == 0
    )
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block
        )
        == 0
    )
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert num_free_blocks == (scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    # NOTE(rob): just the ref count on blocks will be 0. The hash
    # value, etc will remain since we lazily evict for prefix cache.
    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0
        # assert block._block_hash is None
    # assert (
    #     len(scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    #           ) == 0)


def test_memory_leak():
    """Test that we do not have a memory leak."""

    scheduler = create_scheduler(enable_prefix_caching=True)

    NUM_REQUESTS = 5
    NUM_TOKENS = 10
    MAX_TOKENS = 10
    requests = create_requests(
        num_requests=NUM_REQUESTS, num_tokens=NUM_TOKENS, max_tokens=MAX_TOKENS
    )

    # Add each request.
    for request in requests:
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Iterate until done.
    while True:
        scheduler_output = scheduler.schedule()
        if len(scheduler.running) == 0:
            break
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm no memory leak.
    assert_scheduler_empty(scheduler)


def create_scheduler_with_priority(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    use_kv_connector: bool = False,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
    num_speculative_tokens: int | None = None,
    use_ec_connector: bool = False,
    ec_role: str | None = None,
) -> Scheduler:
    """Create scheduler with priority policy enabled.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (False)

    Returns:
      {class}`Scheduler` instance with priority scheduling
    """
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="priority",  # Enable priority scheduling
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = (
        KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )
        if use_kv_connector
        else None
    )

    speculative_config: SpeculativeConfig | None = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens
        )

    ec_transfer_config = (
        ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role=ec_role,
            ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test"},
        )
        if use_ec_connector
        else None
    )

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
        ec_transfer_config=ec_transfer_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
    )


_none_hash_initialized = False


def create_requests_with_priority(
    num_requests: int,
    priorities: list[int],
    arrival_times: list[float] | None = None,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    starting_idx: int = 0,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
):
    """Create requests with specified priorities and arrival times."""
    assert len(priorities) == num_requests
    if arrival_times is not None:
        assert len(arrival_times) == num_requests
    else:
        arrival_times = [float(i) for i in range(num_requests)]

    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        prompt_logprobs=prompt_logprobs,
    )
    requests = []

    if mm_hashes_list is not None:
        # NOTE: allow manual input; some mm items can have the same identifier
        # no. of mm_hashes and mm_positions for each request should be identical
        assert mm_positions is not None, (
            "mm_positions must be provided when mm_hashes_list is provided"
        )
        assert len(mm_hashes_list) == len(mm_positions) == num_requests
        assert [len(h) for h in mm_hashes_list] == [len(p) for p in mm_positions]

        # Since same identifier would imply they are identical encoder output
        # Verify mm items with identical identifier are having mm_position.length
        seen_hashes: dict[str, int] = {}

    if req_ids:
        assert len(req_ids) == num_requests
    else:
        req_ids = [f"{i + starting_idx}" for i in range(num_requests)]

    for i in range(num_requests):
        mm_features = []

        for j, position in enumerate(
            mm_positions[i] if mm_positions is not None else []
        ):
            if mm_hashes_list is not None:
                identifier = mm_hashes_list[i][j]

                # Verify if position length is identical
                position_length = position.length
                if identifier in seen_hashes:
                    assert seen_hashes[identifier] == position_length, (
                        f"mm_hash '{identifier}' has inconsistent position lengths: "
                        f"previously {seen_hashes[identifier]}, now {position_length} "
                        f"at request {i}, position {j}"
                    )
                else:
                    seen_hashes[identifier] = position_length
            else:
                # Unique dummy hash for each mm item
                identifier = f"hash{i}_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

        prompt_token_ids = (
            [starting_idx] * num_tokens
            if same_prompt
            else [i + starting_idx] * num_tokens
        )
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=mm_features if mm_features else None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=arrival_times[i],
            priority=priorities[i],
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def test_priority_scheduling_basic_ordering():
    """Test that requests are scheduled in priority order
    (lower value = higher priority)."""
    scheduler = create_scheduler_with_priority()

    # Create requests with different priorities
    # Priority 0 (highest), 1, 2 (lowest)
    priorities = [2, 0, 1]  # Add in non-priority order
    arrival_times = [1.0, 2.0, 3.0]  # All different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-priority order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify priority order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in priority order:
    # req_1 (priority 0), req_2 (priority 1), req_0 (priority 2)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_arrival_time_tiebreaker():
    """Test that arrival time is used
    as tiebreaker when priorities are equal."""
    scheduler = create_scheduler_with_priority()

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1]  # All same priority
    arrival_times = [3.0, 1.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-arrival order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify arrival time order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in arrival time order:
    # req_1 (1.0), req_2 (2.0), req_0 (3.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_mixed_priority_and_arrival():
    """Test priority scheduling with mixed priorities and arrival times."""
    scheduler = create_scheduler_with_priority()

    # Create requests with mixed priorities and arrival times
    priorities = [2, 1, 1, 0]  # Mixed priorities
    arrival_times = [1.0, 3.0, 2.0, 4.0]  # Mixed arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 4

    # Expected order:
    # 1. req_3 (priority 0, arrival 4.0)
    # 2. req_2 (priority 1, arrival 2.0) - earlier arrival than req_1
    # 3. req_1 (priority 1, arrival 3.0)
    # 4. req_0 (priority 2, arrival 1.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["3", "2", "1", "0"]


# This test had previously been passing due to its use of duplicate
# request ids which resulted in incorrect behavior.
# Now that the duplicate req ids had been fixed it fails and
# investigation is needed into whether the priority scheduling
# preemption logic is working as designed or not.
@pytest.mark.skip("needs investigation")
def test_priority_scheduling_preemption():
    """Test that priority scheduling preempts
    lower priority requests when memory is constrained."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        max_num_seqs=3,  # Allow multiple requests
        max_num_batched_tokens=200,
        num_blocks=6,  # Very limited blocks to force memory pressure
        block_size=16,  # Standard block size
    )

    # Create initial low-priority requests that will consume most memory
    low_priority_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],  # Low priority
        arrival_times=[1.0, 2.0],
        num_tokens=30,  # Large enough to consume significant memory,
        req_ids=["lo1", "lo2"],
    )

    # Add and schedule low priority requests
    for request in low_priority_requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2

    # Simulate model execution to move requests to running state
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in low_priority_requests],
        req_id_to_index={
            req.request_id: i for i, req in enumerate(low_priority_requests)
        },
        sampled_token_ids=[[100] for _ in low_priority_requests],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Verify both requests are running
    assert len(scheduler.running) == 2

    # Now add a high-priority request that requires memory allocation
    # This should trigger preemption due to memory constraints
    high_priority_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],  # High priority
        arrival_times=[3.0],
        num_tokens=30,  # Large enough to require significant memory
        req_ids=["hi1"],
    )[0]

    scheduler.add_request(high_priority_request)

    # Schedule again - this should trigger
    # preemption when trying to allocate memory
    output = scheduler.schedule()

    # Due to the scheduler's design, if preemption happens
    # during running request scheduling,
    # waiting requests won't be scheduled in the same step
    # Let's check if preemption occurred by looking at the waiting queue

    # If preemption happened, we should see requests in the
    # waiting queue
    if len(scheduler.waiting) > 1:  # high priority + preempted request
        # Preemption occurred - verify the high priority request
        # gets scheduled next
        output2 = scheduler.schedule()
        assert len(output2.scheduled_new_reqs) == 1
        # High priority request
        assert output2.scheduled_new_reqs[0].req_id == "hi1"
    else:
        # No preemption needed - all requests fit
        # This is also valid behavior if memory allows
        assert len(output.scheduled_new_reqs) == 1
        # High priority request
        assert output.scheduled_new_reqs[0].req_id == "hi1"


def test_priority_scheduling_no_preemption_when_space_available():
    """Test that preemption doesn't happen
    when there's space for new requests."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=3,  # Allow 3 concurrent requests
        max_num_batched_tokens=200,  # Sufficient token budget
    )

    # Add two low-priority running requests
    low_priority_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=30,
        req_ids=["lo1", "lo2"],
    )

    for request in low_priority_requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in low_priority_requests],
        req_id_to_index={
            req.request_id: i for i, req in enumerate(low_priority_requests)
        },
        sampled_token_ids=[[100] for _ in low_priority_requests],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Add high-priority request
    high_priority_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=30,
        req_ids=["hi1"],
    )[0]

    scheduler.add_request(high_priority_request)

    # Schedule - should not preempt since there's space
    output = scheduler.schedule()

    # Should schedule the new request without preemption
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.running) == 3  # All three requests running
    assert len(scheduler.waiting) == 0  # No requests waiting


def test_priority_scheduling_preemption_victim_selection():
    """Test that the correct victim is selected for
    preemption based on priority and arrival time."""
    # This test verifies the priority-based victim selection logic
    # by checking the waiting queue order after adding requests with different
    # priorities
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Force sequential processing to test priority order
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[3, 2, 0],  # Different priorities: low, medium, high
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_2, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "2"  # Highest priority

    # Verify the waiting queue has the remaining requests in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify priority order
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_1 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["1", "0"]


def test_priority_scheduling_equal_priority_preemption():
    """Test arrival time tiebreaker when requests have equal priority."""
    # This test verifies that arrival time is used as a tiebreaker for equal
    # priorities
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Force sequential processing
    )

    # Create requests with same priority but different arrival times
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[2, 2, 2],  # Same priority
        arrival_times=[3.0, 1.0, 2.0],  # Different arrival times
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should schedule the request with earliest arrival time
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "1"  # Earliest arrival (1.0)

    # Verify the waiting queue has remaining requests in arrival time order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify arrival time order
    waiting_requests = list(scheduler.waiting)

    waiting_arrival_times = [req.arrival_time for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (arrival 2.0) then req_0 (arrival 3.0)
    assert waiting_arrival_times == [2.0, 3.0]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_waiting_queue_order():
    """Test that the waiting queue maintains priority order."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Create multiple requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_3, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "3"

    # Verify waiting queue has remaining requests in priority order
    assert len(scheduler.waiting) == 3

    # Extract requests from waiting queue
    # (it's a heap, so we need to pop to see order)
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be ordered by priority: req_1 (1), req_2 (2), req_0 (3)
    assert waiting_req_ids == ["1", "2", "0"]
    assert waiting_priorities == [1, 2, 3]


def test_priority_scheduling_fcfs_fallback():
    """Test that FCFS behavior is maintained when all
    requests have same priority."""
    scheduler = create_scheduler_with_priority()

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1, 1]  # All same priority
    arrival_times = [4.0, 1.0, 3.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule
    output = scheduler.schedule()

    # Should schedule all requests in arrival time order
    assert len(output.scheduled_new_reqs) == 4
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]

    # Expected order by arrival time:
    # req_1 (1.0), req_3 (2.0), req_2 (3.0), req_0 (4.0)
    assert scheduled_req_ids == ["1", "3", "2", "0"]


def test_priority_scheduling_with_limited_slots():
    """Test priority scheduling when max_num_seqs limits concurrent requests."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,  # Only allow 2 concurrent requests
        max_num_batched_tokens=1000,  # Plenty of token budget
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the 2 highest priority requests
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2

    # Should schedule req_3 (priority 0) and req_1 (priority 1)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert "3" in scheduled_req_ids  # Priority 0
    assert "1" in scheduled_req_ids  # Priority 1

    # Remaining requests should be in waiting queue in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify order
    waiting_requests = list(scheduler.waiting)
    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_heap_property():
    """Test that the waiting queue maintains heap
    property for priority scheduling."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Add requests in random priority order
    priorities = [5, 1, 8, 3, 2, 7, 4, 6]
    arrival_times = [float(i) for i in range(len(priorities))]
    requests = create_requests_with_priority(
        num_requests=len(priorities),
        priorities=priorities,
        arrival_times=arrival_times,
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule one request at a time and verify priority order
    scheduled_priorities = []

    while scheduler.waiting:
        output = scheduler.schedule()
        if output.scheduled_new_reqs:
            req = output.scheduled_new_reqs[0]
            scheduled_priorities.append(requests[int(req.req_id)].priority)

            # Simulate completion to make room for next request
            model_output = ModelRunnerOutput(
                req_ids=[req.req_id],
                req_id_to_index={req.req_id: 0},
                sampled_token_ids=[[100]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
            scheduler.update_from_output(output, model_output)

            # Finish the request to make room for the next one
            scheduler.finish_requests(req.req_id, RequestStatus.FINISHED_STOPPED)

    # Verify requests were scheduled in priority order (lowest value first)
    expected_priorities = sorted(priorities)
    assert scheduled_priorities == expected_priorities


def test_schedule_skip_tokenizer_init():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    requests = create_requests(num_requests=5)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)


def test_schedule_skip_tokenizer_init_structured_output_request():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    structured_outputs_params = StructuredOutputsParams(regex="[0-9]+")
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=16,
        structured_outputs=structured_outputs_params,
    )
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1


@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_priority_scheduling_preemption_and_resumption_when_out_of_kv(
    use_ec_connector, ec_role
):
    """Test that priority scheduling preempts lower priority requests
    when out of KV cache space."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,  # Allow multiple requests
        max_num_batched_tokens=200,
        num_blocks=5,  # Can hold 64 tokens (first block is null)
        block_size=16,  # Standard block size
        use_kv_connector=True,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
    )

    # Create a request and schedule it
    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=30,
        starting_idx=0,
    )[0]
    scheduler.add_request(request_low)
    # 1st schedule
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Simulate model execution - 1st decode
    model_output = ModelRunnerOutput(
        req_ids=[request_low.request_id],
        req_id_to_index={request_low.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Create a high priority request and schedule it
    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=32,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    # 2nd schedule
    output = scheduler.schedule()
    # KV cache should be full at this point
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 2

    # Simulate model execution - 2nd decode
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100] for _ in requests],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # 3rd schedule - this should trigger preemption
    # req_low needs 32 tokens = 2 blocks
    # req_high needs 33 tokens = 3 blocks
    # so doesn't fit in 4 blocks.
    output = scheduler.schedule()

    # Should have preempted req_low
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_cached_reqs.req_ids[0] == request_high.request_id
    assert scheduler.requests[request_low.request_id].status == RequestStatus.PREEMPTED
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1

    # Simulate model execution - 3rd decode
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[], [100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    # Finish the requests to make room for the preempted requests to resume
    scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(request_high.request_id, RequestStatus.FINISHED_STOPPED)

    # 4th Schedule - this should trigger the resumption
    output = scheduler.schedule()
    scheduled_cached_reqs = output.scheduled_cached_reqs

    assert len(output.scheduled_new_reqs) == 0
    assert scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Preempted request resumed in scheduled_cached_reqs
    assert len(scheduled_cached_reqs.resumed_req_ids) == 1
    assert len(scheduled_cached_reqs.all_token_ids) == 1
    assert scheduled_cached_reqs.req_ids[0] == request_low.request_id
    assert request_low.request_id in scheduled_cached_reqs.resumed_req_ids
    assert request_low.request_id in scheduled_cached_reqs.all_token_ids
    # Resumed tokens include 30 prompt tokens and 2 decoded tokens
    assert len(scheduled_cached_reqs.all_token_ids[request_low.request_id]) == 32
    assert scheduled_cached_reqs.all_token_ids[request_low.request_id][31] == 100


@pytest.mark.parametrize(
    ("enable_chunked_prefill", "is_encoder_decoder", "expect_enabled"),
    [
        (True, False, True),
        (False, False, False),
        # Encoder-decoder models should always have it disabled
        (False, True, False),
        (True, True, False),
    ],
)
def test_chunked_prefill_disabled_for_encoder_decoder(
    enable_chunked_prefill: bool, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate that chunked prefill is appropriately disabled for
    encoder-decoder models."""
    scheduler_config = SchedulerConfig(
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=is_encoder_decoder,
        # Must <= max_num_batched_tokens if chunked prefill is disabled
        max_model_len=SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )

    # `is_encoder_decoder` should only be used during construction
    # of the config, and otherwise stored in the model config.
    assert "is_encoder_decoder" not in vars(scheduler_config)
    assert "is_encoder_decoder" not in [
        f.name for f in dataclasses.fields(scheduler_config)
    ]
    _validate_chunked_prefill_settings_for_encoder_decoder(
        scheduler_config, is_encoder_decoder, expect_enabled
    )

    # Ensure it is retained in VllmConfig, even after its post-init.
    vllm_config = VllmConfig(scheduler_config=scheduler_config)
    _validate_chunked_prefill_settings_for_encoder_decoder(
        vllm_config.scheduler_config, is_encoder_decoder, expect_enabled
    )


def _validate_chunked_prefill_settings_for_encoder_decoder(
    scheduler_config: SchedulerConfig, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate chunked prefill settings in the scheduler config for
    encoder-decoder models."""
    assert scheduler_config.enable_chunked_prefill is expect_enabled
    if is_encoder_decoder:
        # Encoder-decoder models should automatically disable chunked multimodal
        # inputs as well
        assert scheduler_config.disable_chunked_mm_input is not expect_enabled
    if is_encoder_decoder and not expect_enabled:
        assert scheduler_config.long_prefill_token_threshold == 0


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests start
# NOTE: In E->P->D disagg case, both KV and EC Connector works in P instance
# Unless specify, the existence of KV Connector should not affect any test results
# ==============================================================================


def _assert_right_encoder_cache_allocated(
    scheduler: Scheduler,
    hashes_to_check: list[str] | None = None,
    requests: list[Request] | None = None,
    expected_total_allocated: int | None = None,
):
    """Check whether encoder cache is allocated correctly."""
    encoder_cache_manager = scheduler.encoder_cache_manager

    # Verify encoder cache manager exists
    assert encoder_cache_manager is not None, "Encoder cache manager should exist"

    # Verify number of cache
    if expected_total_allocated is not None:
        assert len(encoder_cache_manager.cached) == expected_total_allocated
        if expected_total_allocated == 0:
            return

    # Verify each request with MM data is in cache
    cached_hashes = set(encoder_cache_manager.cached.keys())

    if hashes_to_check:
        missed_hashes = set(hashes_to_check) - cached_hashes
        assert not missed_hashes, (
            f"Miss hashes: {missed_hashes} "
            f"Existing encoder cache: {encoder_cache_manager.cached}"
        )

    for req in requests if requests is not None else []:
        if req.mm_features:
            mm_hashes = [f.identifier for f in req.mm_features]
            req_hashes = set(mm_hashes)  # unique hashes set
            missed_hashes = req_hashes - cached_hashes
            assert not missed_hashes, (
                f"Miss hashes in cache for request {req.request_id}: {missed_hashes} "
                f"Existing encoder cache: {encoder_cache_manager.cached}"
            )


def _assert_right_ec_connector_metadata(
    output: SchedulerOutput,
    mm_features_list: list[MultiModalFeatureSpec],
):
    """Verify that ECConnector metadata EXACTLY matches the input MM data"""
    # Get the connector metadata
    metadata = output.ec_connector_metadata

    # Create lookup dictionaries for efficient access
    metadata_dict = {mm_data.mm_hash: mm_data for mm_data in metadata.mm_datas}

    # Check all required identifiers exist in metadata; and no extra
    # In ECExampleConnector format
    # NOTE: even having same identifier, the mm_features can be different
    # since their mm_position can be in different offsets, etc
    identifiers_dict = {f.identifier for f in mm_features_list}
    assert set(metadata_dict.keys()) == identifiers_dict

    # Verify the info matches
    for i, mm_feature in enumerate(mm_features_list):
        identifier = mm_feature.identifier
        assert metadata_dict[identifier].mm_hash == identifier
        assert metadata_dict[identifier].num_token == mm_feature.mm_position.length


def _assert_right_encoder_inputs(
    output: SchedulerOutput,
    check_exist: bool | None = True,
    requests: list[Request] | None = None,
    expected_encoder_inputs: list[list[int]] | None = None,
    expected_total_reqs: int | None = None,
):
    """Verify that requests/mm_hashes should (not) in scheduled encoder input
    If check_exist is False, this function returns True
    if requests are NOT in encoder inputs"""

    # Get the scheduled encoder inputs
    # NOTE: scheduled_encoder_inputs is a dictionary with request id as key
    scheduled_encoder_inputs = output.scheduled_encoder_inputs

    # Check if scheduled_encoder_inputs is empty as expected
    if expected_total_reqs is not None:
        assert len(scheduled_encoder_inputs) == expected_total_reqs
        if expected_total_reqs == 0:
            return

    # Number of expected enocder inputs should match number of requests
    if expected_encoder_inputs:
        assert check_exist and requests is not None  # only support expect input exist
        assert len(requests) == len(expected_encoder_inputs)

    # Check request (not) exist as expected
    for i, request in enumerate(requests if requests is not None else []):
        assert (request.request_id in scheduled_encoder_inputs) is check_exist, (
            f"Request {request.id} presence mismatch: expected {check_exist}, "
            f"got {request.id in scheduled_encoder_inputs}"
        )
        if expected_encoder_inputs:
            scheduled_encoder_input = scheduled_encoder_inputs[request.request_id]
            assert scheduled_encoder_input == expected_encoder_inputs[i]


def test_scheduler_no_ec_connector_by_default():
    """Test scheduler doesn't have EC connector by default."""
    scheduler = create_scheduler()
    assert scheduler.ec_connector is None


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_text_only_request(use_kv_connector):
    """Test text-only requests don't allocate encoder cache."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    NUM_PROMPT_TOKENS = 100

    # Create text-only request (no mm_positions)
    requests = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
    )
    assert not requests[0].mm_features  # No MM data

    scheduler.add_request(requests[0])
    output = scheduler.schedule()

    # Should schedule
    assert len(output.scheduled_new_reqs) == 1

    # Scheduled tokens should equal prompt tokens exactly
    scheduled = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled == NUM_PROMPT_TOKENS, (
        f"Text-only should schedule {NUM_PROMPT_TOKENS}, got {scheduled}"
    )

    # Encoder cache should be empty
    _assert_right_encoder_cache_allocated(scheduler, expected_total_allocated=0)

    # ECConnector should carry no metadata
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_cache_hit_external_load(use_kv_connector):
    """Test ec_consumer loads from external cache when hit.
    A normal basic operation for EPD disaggrgation"""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        # kv connector should not effect test results
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Create MM request
    NUM_TOKENS = 200  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS = 100
    mm_hashes_list = [["hash_test1"]]
    mm_positions = [[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS)]]

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=mm_hashes_list,
        mm_positions=mm_positions,
    )[0]

    # Mock cache hit - encoder cache has_exists externally
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request)
    output = scheduler.schedule()
    # Should schedule prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[request.request_id]
    assert scheduled_tokens == NUM_TOKENS

    # Should called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request, 0)

    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request])

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(output, mm_features_list=request.mm_features)

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_cache_miss_computes_locally(use_kv_connector):
    """Test consumer can compute encoder locally when cache miss (fallback)."""
    # encoder cache itself if it doesn't receive it from external storage

    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Verify consumer role
    assert scheduler.ec_connector is not None
    assert not scheduler.ec_connector.is_producer

    # Create MM request
    request_mm_missed = create_requests(
        num_requests=1,
        num_tokens=200,  # Total (including 100 MM)
        mm_positions=[[PlaceholderRange(offset=0, length=100)]],  # 100 MM tokens
    )[0]

    # Mock cache miss - encoder cache doesn't exist externally
    scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    scheduler.add_request(request_mm_missed)
    output = scheduler.schedule()

    # SCHEDULER should decide to compute encoder locally (fallback)
    assert len(output.scheduled_new_reqs) == 1

    # Should schedule full prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[request_mm_missed.request_id]
    assert scheduled_tokens == 200, (
        f"Expected 200 tokens on cache miss, got {scheduled_tokens}"
    )

    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_mm_missed])

    # ECConnector should carry no metadata (missed cache)
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input contain mm for request_mm_missed
    _assert_right_encoder_inputs(
        output,
        requests=[request_mm_missed],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item
        expected_total_reqs=1,
    )

    # Then MODEL_RUNNER will execute the encoder and cache the result


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_with_partial_cache_hit_multi_round(use_kv_connector):
    """Test consumer with partial cache hit (local & connector) with 2 requests."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Create MM request
    NUM_TOKENS_1 = 300  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS_1 = 50
    mm_hashes_list_1 = [["hash1_A", "hash1_B", "hash1_A", "hash1_F"]]
    mm_positions_1 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=100, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=200, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=250, length=NUM_ENCODER_TOKENS_1),
        ]
    ]
    has_cache_item_result_map_1 = {"hash1_A": False, "hash1_B": True, "hash1_F": True}
    # Create request with 4 MM items, with 2 identical items
    request1 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_1,
        mm_hashes_list=mm_hashes_list_1,
        mm_positions=mm_positions_1,
        max_tokens=1,  # For simplicity
    )[0]

    # Mock partial cache hit: 1st and 3rd missing, 2nd and 4th exist
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_val: has_cache_item_result_map_1[hash_val]
    )
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request1)
    output = scheduler.schedule()

    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request1.request_id]
    assert scheduled_tokens == NUM_TOKENS_1

    # Encoder cache should contain all mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request1])

    # Should have called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called()
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for 2nd and 4th mm item
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request1.mm_features[1], request1.mm_features[3]]
    )

    # Should schedule ONLY 1 encoder input (index 0), no repeat for identical items
    _assert_right_encoder_inputs(
        output,
        requests=[request1],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item ONLY
        expected_total_reqs=1,
    )

    # Simulate model execution 1 step
    model_output = ModelRunnerOutput(
        req_ids=[request1.request_id],
        req_id_to_index={request1.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # request1 is finished after outputing 1 token
    # Finish request
    scheduler.finish_requests(request1.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)

    # Create another request with 4 MM items
    NUM_TOKENS_2 = 400
    NUM_ENCODER_TOKENS_2 = 50
    mm_hashes_list_2 = [["hash1_C", "hash1_D", "hash1_E", "hash1_A"]]
    mm_positions_2 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=100, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=200, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=250, length=NUM_ENCODER_TOKENS_2),
        ]
    ]
    has_cache_item_result_map_2 = {
        "hash1_C": True,
        "hash1_D": False,
        "hash1_E": False,
        "hash1_A": True,
    }
    request2 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_2,
        mm_hashes_list=mm_hashes_list_2,
        mm_positions=mm_positions_2,
        max_tokens=1,  # For simplicity
    )[0]

    # Mock partial cache hit: only hash1_A and hash1_C exist in connector
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_val: has_cache_item_result_map_2[hash_val]
    )

    scheduler.add_request(request2)
    output = scheduler.schedule()

    # Check
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request2.request_id]
    assert scheduled_tokens == 400

    # Encoder cache should contain all mm items from request2
    _assert_right_encoder_cache_allocated(scheduler, requests=[request2])

    # Should call update_state_after_alloc for hash1_C, ONLY
    # hash1_A should not be loaded from connector
    # since it's computed in last request & exist in local cache
    # Order of getting encoder cache should be: local cache -> connector-> compute
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 0)
    scheduler.ec_connector.update_state_after_alloc.assert_called_once()

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for hash1_C only (index 0)
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[0]]
    )

    # Should schedule 2 encoder input hash1_D and hash1_E (index 1, 2)
    _assert_right_encoder_inputs(
        output,
        requests=[request2],
        expected_encoder_inputs=[[1, 2]],
        expected_total_reqs=1,
    )


@pytest.mark.parametrize("cache_exist", ["local", "connector_only", "no_where"])
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_schedule_multiple_requests(cache_exist, use_kv_connector):
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=10,  # allow multiple requests
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    mm_hashes_list = [[f"hash_{i}"] for i in range(10)]
    mm_positions = [[PlaceholderRange(offset=i, length=100)] for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_hashes_list=mm_hashes_list,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    # Set up to test different encoder cache exsistence scenario after preemption
    # Order of getting encoder cache should be: local cache -> connector-> compute
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    if cache_exist == "local":
        # Allocate cache to cache manager manually to mimick
        for req in requests:
            scheduler.encoder_cache_manager.allocate(req, 0)
    else:
        # Make sure local encoder cache empty
        scheduler.encoder_cache_manager.cached = {}

    if cache_exist == "connector_only":
        # Cache exist in ec_connector
        scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    elif cache_exist == "no_where":
        scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    ## Encoder-cache-specific checks:
    # mm_hashes of requests exist in cache after scheduling for all scenario
    _assert_right_encoder_cache_allocated(scheduler, requests=requests)

    # Should only call update_state_after_alloc when loaded externally
    if cache_exist == "connector_only":
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            requests[-1], 0
        )

        # Concat mm_features for the 10 requests together
        mm_features_list = [feature for req in requests for feature in req.mm_features]

        # Check metadata should contain mm data for all 10 requests
        _assert_right_ec_connector_metadata(output, mm_features_list=mm_features_list)
    else:
        scheduler.ec_connector.update_state_after_alloc.assert_not_called()
        # ECConnector should carry no metadata
        _assert_right_ec_connector_metadata(output, mm_features_list=[])

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # Should only schedule encoder input when cache is not found anywhere
    if cache_exist == "no_where":
        _assert_right_encoder_inputs(
            output,
            requests=requests,
            expected_encoder_inputs=[[0] for _ in range(10)],
            expected_total_reqs=10,
        )
    else:
        _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_unable_to_allocate(use_kv_connector):
    """
    Test whether scheduler with ECConnector is able to handle
    unable to allocate (run out of blocks).
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Mock ec_connector load external cache behavior
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=[["hash_1"], ["hash_2"]],
        mm_positions=[
            [PlaceholderRange(offset=1, length=10)] for _ in range(NUM_REQUESTS)
        ],
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Setup MODEL_RUNNER_OUTPUT to be run in _step_until_done later
    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Just one request should be running.
    output = scheduler.schedule()
    scheduled_tokens = output.num_scheduled_tokens[scheduler.running[0].request_id]
    assert scheduled_tokens == NUM_TOKENS
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Should have called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(
        scheduler.running[0], 0
    )
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    scheduled_tokens = output.num_scheduled_tokens[scheduler.running[0].request_id]
    assert scheduled_tokens == NUM_TOKENS
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # update_state_after_alloc should be called for loading external cache
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(
        scheduler.running[0], 0
    )
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


@pytest.mark.parametrize("cache_exist", ["local", "connector_only", "no_where"])
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_priority_scheduling_ec_connector_preemption_and_resumption(
    cache_exist, use_kv_connector
):
    """Test that priority scheduling preempts lower priority requests
    when out of KV cache space."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        max_num_seqs=2,  # allow multiple requests
        # kv connector should not effect test results
        use_kv_connector=use_kv_connector,
        num_blocks=15,  # can hold 244 tokens with 14 blocks (first block is null)
        block_size=16,  # standard block size
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Mock cache hit: Both cache exist in connector (at E->PD initially)
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    # Create a request and schedule it (and to be preempted)
    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=94,
        mm_hashes_list=[["hash_low"]],
        # NOTE: this test only preempt the last block.
        # Setting mm_position at the last block can force to recompute encoding
        mm_positions=[[PlaceholderRange(offset=82, length=10)]],
        starting_idx=0,
    )[0]
    scheduler.add_request(request_low)
    # 1st schedule
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    scheduled_tokens = output.num_scheduled_tokens[request_low.request_id]
    assert scheduled_tokens == 94
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    ## Encoder-cache-specific checks:
    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_low])

    # Verify update_state_after_alloc called (external load)
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request_low, 0)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_low.mm_features
    )

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 1st decode
    model_output = ModelRunnerOutput(
        req_ids=[request_low.request_id],
        req_id_to_index={request_low.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Create a high priority request and schedule it
    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=128,
        mm_hashes_list=[["hash_high"]],
        mm_positions=[[PlaceholderRange(offset=1, length=10)]],
        max_tokens=2,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    # 2nd schedule
    output = scheduler.schedule()

    # KV cache should be full at this point
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 2

    ## Encoder-cache-specific checks:
    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_high])

    # Verify update_state_after_alloc called (external load)
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request_high, 0)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_high.mm_features
    )

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 2nd decode
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100] for _ in requests],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # 3rd schedule - - this should trigger preemption
    # req_low needs 96 tokens = 6 blocks
    # req_high needs 129 tokens = 9 blocks
    # so doesn't fit in 14 blocks.
    output = scheduler.schedule()

    # Should have preempted req_low
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_cached_reqs.req_ids[0] == request_high.request_id
    assert scheduler.requests[request_low.request_id].status == RequestStatus.PREEMPTED
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1

    ## Encoder-cache-specific checks:
    # request_high is in decode phase now
    # ECConnector should carry no metadata
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 3rd decode, after req_low was preempted
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100], [100, 200]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    # Finish the requests to make room for the preempted requests to resume
    # req_high is finished after outputing 2 tokens
    scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(
        request_high.request_id, RequestStatus.FINISHED_LENGTH_CAPPED
    )

    # Set up to test different encoder cache exsistence scenario after preemption
    # Order of getting encoder cache should be: local cache -> connector-> compute
    # By default, the cache should still exist in local in this test case
    if cache_exist != "local":
        # Make local encoder cache empty
        scheduler.encoder_cache_manager.cached = {}

    if cache_exist == "connector_only":
        # Cache exist in ec_connector
        scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    elif cache_exist == "no_where":
        scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    # 4th Schedule - this should trigger req_low resumption from waiting
    output = scheduler.schedule()
    scheduled_cached_reqs = output.scheduled_cached_reqs

    assert len(output.scheduled_new_reqs) == 0
    assert scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Preempted request resumed in scheduled_cached_reqs
    assert len(scheduled_cached_reqs.resumed_req_ids) == 1
    assert len(scheduled_cached_reqs.all_token_ids) == 1
    assert scheduled_cached_reqs.req_ids[0] == request_low.request_id
    assert request_low.request_id in scheduled_cached_reqs.resumed_req_ids
    assert request_low.request_id in scheduled_cached_reqs.all_token_ids
    ## Resumed tokens include 94 prompt tokens and 2 decoded tokens
    assert len(scheduled_cached_reqs.all_token_ids[request_low.request_id]) == 96
    assert scheduled_cached_reqs.all_token_ids[request_low.request_id][95] == 100
    assert scheduler.running[0].request_id == request_low.request_id
    assert request_high.request_id in output.finished_req_ids

    ## Encoder-cache-specific checks:
    # mm_hash of request_low exists in cache after scheduling for all scenario
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_low])

    # Should only call update_state_after_alloc when loaded externally
    if cache_exist == "connector_only":
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            request_low, 0
        )
        _assert_right_ec_connector_metadata(
            output, mm_features_list=request_low.mm_features
        )
    else:
        scheduler.ec_connector.update_state_after_alloc.assert_not_called()
        # ECConnector should carry no metadata
        _assert_right_ec_connector_metadata(output, mm_features_list=[])

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # Should only schedule encoder input when cache is not found anywhere
    if cache_exist == "no_where":
        _assert_right_encoder_inputs(
            output,
            requests=[request_low],
            expected_encoder_inputs=[[0]],
            expected_total_reqs=1,
        )
    else:
        _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_allocate_encoder_tokens_with_external_load(use_kv_connector):
    """
    Scenario:
      - Encoder cache size: 32
      - Request A: 1 feature (12 tokens) → NOT cached remotely.
      - Request B: 3 features (3 x 10 tokens) → ALL cached remotely.

    Steps:
      1. Schedule Request A (locally uses 12 tokens).
      2. Schedule Request B (remote cache) - only schedule 1st and 2nd
      3. Free A's cache, then schedule B again (continuation) - schedule 3rd image
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        block_size=16,
        num_blocks=11,  # Can hold 160 tokens (first block is null)
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Limit the number of availiable slots of EncoderCacheManager
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=32)

    # Create MM request1
    NUM_TOKENS_1 = 50  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS_1 = 12
    mm_hashes_list_1 = [["hash1_1"]]
    mm_positions_1 = [[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_1)]]

    request1 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_1,
        mm_hashes_list=mm_hashes_list_1,
        mm_positions=mm_positions_1,
        max_tokens=1,  # For simplicity
        req_ids=["req1"],
    )[0]

    # Create MM request1 with 3 MM items
    NUM_TOKENS_2 = 40
    NUM_ENCODER_TOKENS_2 = 10
    mm_hashes_list_2 = [["hash2_1", "hash2_2", "hash2_3"]]
    mm_positions_2 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=12, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=24, length=NUM_ENCODER_TOKENS_2),
        ]
    ]

    request2 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_2,
        mm_hashes_list=mm_hashes_list_2,
        mm_positions=mm_positions_2,
        max_tokens=10,
        req_ids=["req2"],
    )[0]

    # Mock cache hit: MM of request1 NOT cached remotely, request2 cached remotely
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_value: hash_value in mm_hashes_list_2[0]
    )
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request1)
    scheduler.add_request(request2)
    output = scheduler.schedule()

    # Now, since encoder cache manager can only store 32 tokens
    # It should allocated mm item hash1_1, hash2_1 and hash2_2
    scheduled_tokens = output.num_scheduled_tokens[request1.request_id]
    assert scheduled_tokens == NUM_TOKENS_1
    assert scheduler.get_num_unfinished_requests() == 2

    # Encoder cache should contain mm item from request1
    _assert_right_encoder_cache_allocated(
        scheduler, hashes_to_check=["hash1_1", "hash2_1", "hash2_2"]
    )

    # request2's 2nd mm item is the last call of update_state_after_alloc
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 1)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of hash2_1 and hash2_2 ONLY
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[0], request2.mm_features[1]]
    )

    # Should schedule ONLY 1 encoder input
    _assert_right_encoder_inputs(
        output,
        requests=[request1],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item of request1
        expected_total_reqs=1,
    )

    # Simulate model execution 1 step
    model_output = ModelRunnerOutput(
        req_ids=[request1.request_id, request2.request_id],
        req_id_to_index={request1.request_id: 0, request2.request_id: 1},
        sampled_token_ids=[[100], [121]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # request1 is finished after outputing 1 token
    # Finish request
    scheduler.finish_requests(request1.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)
    assert scheduler.get_num_unfinished_requests() == 1

    # Schedule again; Now request1's encoder cache should be freed
    # -> hash2_3 can be scheduled and allocated
    output = scheduler.schedule()

    # Check
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request2.request_id]
    print(f"Hero: scheduled_tokens for req2: {scheduled_tokens}")
    print(f"hero: num_scheduled_tokens 2: {output.num_scheduled_tokens}")

    # Encoder cache should contain all mm items from request2
    _assert_right_encoder_cache_allocated(scheduler, requests=[request2])

    # request2's 3rd mm item is the ONLY call of update_state_after_alloc
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 2)
    scheduler.ec_connector.update_state_after_alloc.assert_called_once()

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for hash2_3 ONLY
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[2]]
    )

    # Should schedule no encoder input
    _assert_right_encoder_inputs(
        output,
        expected_total_reqs=0,
    )


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests end
# ==============================================================================


def test_prepend_skipped_requests_order():
    scheduler = create_scheduler(max_num_seqs=1, use_kv_connector=True)
    requests = create_requests(num_requests=4)
    for request in requests:
        scheduler.add_request(request)

    # 4 requests waiting, capture their order
    expected_waiting_reqs = list(scheduler.waiting)

    # simulate first 2 waiting requests are waiting for remote KVs
    for req in expected_waiting_reqs[:2]:
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS

    # schedule step
    # expect the first 2 waiting to be skipped, the third running,
    # and the fourth waiting
    scheduler.schedule()

    # pop the third request which is expected to be running
    expected_waiting_reqs.pop(2)

    # verify waiting order is preserved
    assert list(scheduler.waiting) == expected_waiting_reqs


def test_abort_request_waiting_for_remote_kvs():
    scheduler = create_scheduler(use_kv_connector=True)

    # add a single request
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    # set request to waiting for remote KVs, and abort it
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finish_requests((request.request_id,), RequestStatus.FINISHED_ABORTED)
    assert request.status == RequestStatus.FINISHED_ABORTED

    # verify request is not deleted
    assert request.request_id in scheduler.requests

    # finish recving request
    scheduler_output = scheduler.schedule()
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_connector_output=KVConnectorOutput(finished_recving={request.request_id}),
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # assert request is deleted
    assert request.request_id not in scheduler.requests
    assert not scheduler.finished_recving_kv_req_ids


def test_abort_request_finished_recving():
    scheduler = create_scheduler(use_kv_connector=True)

    # add a single request
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    # set request to waiting for remote KVs, finished but not yet updated
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finished_recving_kv_req_ids.add(request.request_id)

    # abort request
    scheduler.finish_requests((request.request_id,), RequestStatus.FINISHED_ABORTED)
    assert request.status == RequestStatus.FINISHED_ABORTED

    # verify request is deleted
    assert request.request_id not in scheduler.requests
    assert not scheduler.finished_recving_kv_req_ids
