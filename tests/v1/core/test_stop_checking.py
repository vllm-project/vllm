from typing import List

import pytest

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus, SamplingParams

EOS_TOKEN_ID = 50256


@pytest.fixture
def scheduler():
    cache_config = CacheConfig(block_size=16,
                               gpu_memory_utilization=0.9,
                               swap_space=0.1,
                               cache_dtype="auto")
    cache_config.num_gpu_blocks = 100
    return Scheduler(scheduler_config=SchedulerConfig(),
                     model_config=ModelConfig(model="facebook/opt-125m",
                                              task="auto",
                                              tokenizer="test_tokenizer",
                                              tokenizer_mode="auto",
                                              trust_remote_code=False,
                                              dtype="float16",
                                              seed=42),
                     cache_config=cache_config,
                     lora_config=None)


def _create_test_request(request_id: str, max_tokens: int,
                         stop_token_ids: List[int]) -> Request:
    return Request(request_id=request_id,
                   prompt="test prompt",
                   prompt_token_ids=[1, 2, 3],
                   multi_modal_inputs=None,
                   multi_modal_hashes=None,
                   multi_modal_placeholders=None,
                   sampling_params=SamplingParams(
                       max_tokens=max_tokens, stop_token_ids=stop_token_ids),
                   eos_token_id=EOS_TOKEN_ID,
                   arrival_time=0.0)


def test_multiple_stop_tokens(scheduler):
    """Test with stop when generating multiple tokens"""
    # Nonstop case
    request = _create_test_request("test1", 100, stop_token_ids=[42, 43, 44])
    scheduler.requests[request.request_id] = request
    request.append_output_token_ids([4, 5, 6, 7, 8])
    result = scheduler._check_stop(request)
    assert result is False

    # EOS token is generated in the beginning of the output tokens
    request = _create_test_request("test1", 100, stop_token_ids=[42, 43, 44])
    scheduler.requests[request.request_id] = request
    request.append_output_token_ids([EOS_TOKEN_ID, 5, EOS_TOKEN_ID, 7, 43, 5])
    result = scheduler._check_stop(request)
    assert result is True
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.request_id in scheduler.finished_req_ids
    # Should be cropped at the first stop token
    assert len(request.output_token_ids) == 1
    assert list(request.output_token_ids) == [EOS_TOKEN_ID]

    # Stop token, 43 is one of the stop tokens
    request = _create_test_request("test1", 100, stop_token_ids=[42, 43, 44])
    scheduler.requests[request.request_id] = request
    request.append_output_token_ids([4, 5, 43, 7, 43, 5])
    result = scheduler._check_stop(request)
    assert result is True
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == 43
    assert request.request_id in scheduler.finished_req_ids
    # Should be cropped at the first stop token
    assert len(request.output_token_ids) == 3
    assert list(request.output_token_ids) == [4, 5, 43]

    # Max tokens, should be cropped when reaching the max tokens
    max_tokens = 2
    request = _create_test_request("test2",
                                   max_tokens,
                                   stop_token_ids=[42, 43, 44])
    scheduler.requests[request.request_id] = request
    output_token_ids = [4, 5, 43, 7, 43, 5]
    request.append_output_token_ids(output_token_ids)
    result = scheduler._check_stop(request)
    assert result is True
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.request_id in scheduler.finished_req_ids
    # Should be cropped at the first stop token
    assert len(request.output_token_ids) == max_tokens
    assert list(request.output_token_ids) == output_token_ids[:max_tokens]
