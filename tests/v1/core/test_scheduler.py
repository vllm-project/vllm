# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
) -> Scheduler:
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
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
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    cache_config.num_gpu_blocks = 10000
    return Scheduler(scheduler_config,
                     model_config,
                     cache_config,
                     lora_config=None)


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_positions: Optional[List[PlaceholderRange]] = None,
):
    sampling_params = SamplingParams()
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
            eos_token_id=None,
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


def test_schedule():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
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
        sampled_token_ids=[0] * len(requests),
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
