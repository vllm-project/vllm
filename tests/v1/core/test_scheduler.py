# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: Optional[bool] = None,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    use_kv_connector: bool = False,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: Optional[int] = None,
    num_speculative_tokens: Optional[int] = None,
) -> Scheduler:
    '''Create scheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (None)

    Returns:
      :class:`Scheduler` instance
    '''
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=True,
    )
    model_config = ModelConfig(
        model=model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    # Cache config, optionally force APC
    kwargs_cache = ({} if enable_prefix_caching is None else {
        'enable_prefix_caching': enable_prefix_caching
    })
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        **kwargs_cache,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    ) if use_kv_connector else None

    speculative_config: Optional[SpeculativeConfig] = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens)

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_requests(num_requests: int,
                    num_tokens: int = 10,
                    mm_positions: Optional[list[PlaceholderRange]] = None,
                    max_tokens: int = 16,
                    stop_token_ids: Optional[list[int]] = None,
                    prompt_logprobs: Optional[int] = None):
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs)
    requests = []
    for i in range(num_requests):
        if mm_positions is not None:
            mm_position = mm_positions[i]
            mm_inputs = [MultiModalKwargs({})] * len(mm_position)
        else:
            mm_position = None
            mm_inputs = None
        request = Request(
            request_id=f"{i}",
            prompt=None,
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            multi_modal_inputs=mm_inputs,
            multi_modal_placeholders=mm_position,
            multi_modal_hashes=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0,
        )
        requests.append(request)
    return requests


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
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
    (True, 5),
])
def test_schedule(enable_prefix_caching: Optional[bool],
                  prompt_logprobs: Optional[int]):
    '''Test scheduling. 
    Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
    '''
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10,
                               prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert len(output.scheduled_cached_reqs) == 0
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
    mm_positions = [[PlaceholderRange(offset=i, length=100)]
                    for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)
    assert len(output.scheduled_encoder_inputs) == 10
    for req_id, encoder_input in output.scheduled_encoder_inputs.items():
        assert len(encoder_input) == 1


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
    mm_positions = [[PlaceholderRange(offset=100, length=600)]
                    for _ in range(3)]
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert len(output.scheduled_cached_reqs) == 0
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
    req_to_index = {
        request.request_id: i
        for i, request in enumerate(requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], [], []],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    # Only the first and second requests are scheduled.
    # The third request is in the RUNNING state but not scheduled in this step
    # because of the encoder budget.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 0
    assert len(output.scheduled_cached_reqs) == 2
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
    requests = create_requests(num_requests=1,
                               num_tokens=1200,
                               mm_positions=mm_positions)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0
    # We want to only see the 400 text tokens at the start scheduled
    assert output.num_scheduled_tokens[requests[0].request_id] == 400

    req_to_index = {
        request.request_id: i
        for i, request in enumerate(requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(output.scheduled_new_reqs) == 0
    assert len(output.scheduled_cached_reqs) == 1
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
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[0].request_id] == 400
    # The second request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[1].request_id] == 400
    # The third request is also scheduled partially - 1024 - 400 - 400 = 224.
    assert output.num_scheduled_tokens[requests[2].request_id] == 224
    req_to_index = {
        request.request_id: i
        for i, request in enumerate(requests)
    }
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step. All three requests are running.
    # Processed the remaining prefills of the first and second requests.
    output1 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output1.scheduled_new_reqs) == 0
    assert len(output1.scheduled_cached_reqs) == 3
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
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output1, model_runner_output)
    output2 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output2.scheduled_new_reqs) == 0
    assert len(output2.scheduled_cached_reqs) == 3
    assert len(output2.finished_req_ids) == 0
    assert output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert output2.num_scheduled_tokens[requests[1].request_id] == 1
    assert output2.num_scheduled_tokens[
        requests[2].request_id] == 800 - 224 - 224


def test_stop_via_update_from_output():
    """Test stopping behavior through update_from_output"""
    scheduler = create_scheduler(num_speculative_tokens=1)

    # Test case 1: Stop on EOS token
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 1,
                                           requests[1].request_id: 2
                                       },
                                       total_num_scheduled_tokens=3,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={
                                           requests[0].request_id: [],
                                           requests[1].request_id: [10]
                                       },
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[EOS_TOKEN_ID],
                           [10,
                            11]],  # First request hits EOS, second continues
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

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
    requests = create_requests(num_requests=2,
                               max_tokens=10,
                               stop_token_ids=[42, 43])
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 3,
                                           requests[1].request_id: 2
                                       },
                                       total_num_scheduled_tokens=5,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={
                                           requests[0].request_id: [10, 42],
                                           requests[1].request_id: [13]
                                       },
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[10, 42, 12],
                           [13, 14]],  # First request hits stop token
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

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

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 3,
                                           requests[1].request_id: 1
                                       },
                                       total_num_scheduled_tokens=4,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={
                                           requests[0].request_id: [10, 11],
                                           requests[1].request_id: []
                                       },
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[10, 11, 12],
                           [13]],  # First request exceeds max_tokens
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped due to length
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 11
                                                  ]  # Truncated to max_tokens
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
        scheduled_cached_reqs=[],
        num_scheduled_tokens={requests[0].request_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [EOS_TOKEN_ID, 10]
        },
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify request continues past EOS
    assert len(scheduler.running) == 1
    assert not requests[0].is_finished()
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID, 10, 11]


@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
    (True, 5),
])
def test_schedule_concurrent_batches(enable_prefix_caching: Optional[bool],
                                     prompt_logprobs: Optional[int]):
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
    assert scheduler_output0.num_scheduled_tokens[
        requests[0].request_id] == 512

    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[
        requests[1].request_id] == 512

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
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
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)


# Note - these test cases mirror some of those in test_rejection_sampler.py
@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2, 3]], [[1, 2, 3, 4]], (1, 3, 3, [1, 1, 1])),  # perfect match
        ([[1, 2, 3]], [[1, 5]], (1, 3, 1, [1, 0, 0])),  # early mismatch
        ([[1, 2], [3]], [[1, 2, 5], [3, 4]],
         (2, 3, 3, [2, 1])),  # multiple sequences
        ([[1]], [[1, 2]], (1, 1, 1, [1])),  # single token sequence
        ([[]], [[5]], (0, 0, 0, [0])),  # empty sequence
        ([[1, 2, 3], [4, 5, 6]], [[1, 2, 7], [4, 8]],
         (2, 6, 3, [2, 1, 0])),  # multiple mismatches
    ])
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
        spec_token_ids=spec_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    engine_core_outputs = scheduler.update_from_output(output,
                                                       model_runner_output)

    for i in range(len(requests)):
        running_req = scheduler.running[i]
        # The prompt token
        assert running_req.num_computed_tokens == 1
        # The prompt token and the sampled token
        assert running_req.num_tokens == 2
        # The prompt token, the sampled token, and the speculated tokens
        assert running_req.num_tokens_with_spec == 2 + len(spec_tokens[i])

    # No draft or accepted tokens counted yet
    assert engine_core_outputs.scheduler_stats.spec_decoding_stats is None

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    # The sampled token and speculated tokens
    assert output.total_num_scheduled_tokens == \
        len(requests) + sum(len(ids) for ids in spec_tokens)
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1 + len(spec_tokens[i])
        if spec_tokens[i]:
            assert len(output.scheduled_spec_decode_tokens[req_id]) == \
                len(spec_tokens[i])
        else:
            assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=output_tokens,
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    engine_core_outputs = scheduler.update_from_output(output,
                                                       model_runner_output)

    scheduler_stats = engine_core_outputs.scheduler_stats
    if expected[0] == 0:
        assert scheduler_stats.spec_decoding_stats is None
    else:
        assert scheduler_stats.spec_decoding_stats is not None
        stats = scheduler_stats.spec_decoding_stats
        assert stats.num_drafts == expected[0]
        assert stats.num_draft_tokens == expected[1]
        assert stats.num_accepted_tokens == expected[2]
        assert stats.num_accepted_tokens_per_pos == expected[3]


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
    req_ids: list[str],
    num_tokens: int,
    block_size: int,
    num_requests: int,
    num_total_blocks: int,
):
    """Check whether KVCacheManager is correct after allocate."""

    # Make sure the request stats are right.
    EXPECTED_TOTAL_BLOCKS = num_tokens // block_size
    for req_id in req_ids:
        blocks = scheduler.kv_cache_manager.req_to_blocks[req_id]
        hashes = scheduler.kv_cache_manager.req_to_block_hashes[req_id]
        assert (scheduler.kv_cache_manager.num_cached_block[req_id] ==
                EXPECTED_TOTAL_BLOCKS)
        assert len(blocks) == EXPECTED_TOTAL_BLOCKS
        assert len(hashes) == EXPECTED_TOTAL_BLOCKS

    # Make sure we actually touched all the blocks.
    BLOCKS_PER_REQ = num_tokens / block_size
    assert (scheduler.kv_cache_manager.block_pool.get_num_free_blocks() ==
            num_total_blocks - num_requests * BLOCKS_PER_REQ)


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
        assert len(output.kv_connector_metadata.requests) == 0
        ecos = scheduler.update_from_output(output, model_runner_output)
        all_done = True
        for eco in ecos.outputs:
            if eco.finish_reason is None:
                all_done = False
        all_finished = all_done


def test_kv_connector_basic():
    """
    Test whether Scheduler with KVConnector schedules tokens, allocates
    memory, and cleans up requests as expected under normal operation.
    """

    # Setup Scheduler.
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=True,
    )
    NUM_TOTAL_BLOCKS = (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
    BLOCK_SIZE = scheduler.cache_config.block_size

    # Mock External Cache Hit.
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler.connector.get_num_new_matched_tokens = Mock(name="method")
    scheduler.connector.get_num_new_matched_tokens.return_value = (
        NUM_MATCHED_NEW_TOKENS)

    ######################################################
    # FIRST SET OF REQUESTS - External Hit Only
    NUM_REQUESTS = 2
    NUM_TOKENS = NUM_MATCHED_NEW_TOKENS * 2
    MAX_TOKENS = 3
    requests = create_requests(num_requests=NUM_REQUESTS,
                               num_tokens=NUM_TOKENS,
                               max_tokens=MAX_TOKENS)
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
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
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
    _assert_right_kv_cache_manager(scheduler, req_ids, NUM_TOKENS, BLOCK_SIZE,
                                   NUM_REQUESTS, NUM_TOTAL_BLOCKS)

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() \
        == NUM_TOTAL_BLOCKS

    ######################################################
    # SECOND SET OF REQUESTS - Local And External Hit
    NUM_TOKENS_PREFIX = NUM_TOKENS
    # We will get a local prefix cache hit for the first
    # NUM_TOKENS_PREFIX tokens since they are used above.
    NUM_TOKENS = NUM_TOKENS_PREFIX * 2
    requests = create_requests(num_requests=NUM_REQUESTS,
                               num_tokens=NUM_TOKENS,
                               max_tokens=MAX_TOKENS)
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
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )

    # We should get a local cache hit of NUM_TOKENS_PREFIX and
    # a remote KV cache hit of NUM_MATCHED_NEW_TOKENS.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens after local + remote cache hit.
        expected_num_scheduled_tokens=(NUM_TOKENS - NUM_TOKENS_PREFIX -
                                       NUM_MATCHED_NEW_TOKENS))

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(scheduler, req_ids, NUM_TOKENS, BLOCK_SIZE,
                                   NUM_REQUESTS, NUM_TOTAL_BLOCKS)

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() \
        == NUM_TOTAL_BLOCKS


def test_kv_connector_unable_to_allocate():
    """
    Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=True,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
    )
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler.connector.get_num_new_matched_tokens = Mock(name="method")
    scheduler.connector.get_num_new_matched_tokens.return_value = (
        NUM_MATCHED_NEW_TOKENS)

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(num_requests=NUM_REQUESTS,
                               num_tokens=NUM_TOKENS,
                               max_tokens=MAX_TOKENS)
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
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(output,
                                   num_requests=1,
                                   expected_num_scheduled_tokens=NUM_TOKENS -
                                   NUM_MATCHED_NEW_TOKENS)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() \
        == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(output,
                                   num_requests=1,
                                   expected_num_scheduled_tokens=NUM_TOKENS -
                                   NUM_MATCHED_NEW_TOKENS)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() \
        == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


def test_kv_connector_handles_preemption():
    """
    Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 2
    # NOTE: there is 1 null block, so this is 6 blocks.
    NUM_BLOCKS = 7
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=True,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
    )

    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE
    scheduler.connector.get_num_new_matched_tokens = Mock(name="method")
    scheduler.connector.get_num_new_matched_tokens.return_value = (
        NUM_MATCHED_NEW_TOKENS)

    # Create two requests.
    # Both can be scheduled at first, but the second request
    # will be preempted and re-scheduled.
    NUM_REQUESTS = 2
    NUM_TOKENS = BLOCK_SIZE * 2 + 1
    MAX_TOKENS = BLOCK_SIZE * 2
    requests = create_requests(num_requests=NUM_REQUESTS,
                               num_tokens=NUM_TOKENS,
                               max_tokens=MAX_TOKENS)
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
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )

    # All can be scheduled - 1st token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # 2 remote kv cache hits.
        num_requests=2,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS)
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # All can be scheduled - 2nd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1)
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # This will generate a new block and cause a preemption - 3rd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1)
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
        expected_num_scheduled_tokens=1)
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() \
        == NUM_BLOCKS - 1

    # Restarts the preempted request - generate 3rd token.
    # This will have a local and remote cache hit.
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
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1)
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() \
        == NUM_BLOCKS - 1
