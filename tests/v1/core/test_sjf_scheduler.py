# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test


def create_scheduler_with_sjf(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
) -> Scheduler:
    """Create scheduler with SJF policy enabled.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (False)

    Returns:
      {class}`Scheduler` instance with SJF scheduling
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
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="sjf",  # Enable SJF scheduling
    )

    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size, 1, 1, torch.float32, False)
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


def create_requests_for_sjf(
    num_requests: int,
    prompt_lengths: list[int],
    arrival_times: list[float] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    starting_idx: int = 0,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
):
    """Create requests with specified prompt lengths and arrival times for SJF testing."""
    assert len(prompt_lengths) == num_requests
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

    if req_ids:
        assert len(req_ids) == num_requests
    else:
        req_ids = [f"{i + starting_idx}" for i in range(num_requests)]

    for i in range(num_requests):
        num_tokens = prompt_lengths[i]
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
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=arrival_times[i],
            priority=1,  # SJF ignores priority, set to default
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def test_sjf_scheduling_basic_ordering():
    """Test that requests are scheduled in SJF order
    (shorter job = higher priority)."""
    scheduler = create_scheduler_with_sjf()

    # Create requests with different prompt lengths
    # Shorter jobs should be scheduled first
    prompt_lengths = [100, 50, 75]  # Add in non-length order
    arrival_times = [0.0, 0.0, 0.0]  # All same arrival times
    requests = create_requests_for_sjf(
        num_requests=3, prompt_lengths=prompt_lengths, arrival_times=arrival_times
    )

    # Add requests in non-length order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify SJF order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in length order (shortest first):
    # req_1 (length 50), req_2 (length 75), req_0 (length 100)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_sjf_scheduling_waiting_time_tiebreaker_fixed():
    """Test that waiting time is used as tiebreaker when lengths are equal.
    """
    scheduler = create_scheduler_with_sjf()
    
    # Mock current time, fixed at 10.0 seconds
    current_time = 10.0
    time_patch = Mock(return_value=current_time)
    
    with patch('time.time', time_patch):
        # Create 3 requests with same length but different arrival times
        prompt_lengths = [64, 64, 64]  # All requests have same length
        # Arrival times: req1 earliest, req2 second, req0 latest
        arrival_times = [3.0, 1.0, 2.0]
        
        requests = create_requests_for_sjf(
            num_requests=3, 
            prompt_lengths=prompt_lengths, 
            arrival_times=arrival_times
        )

        # Add requests to scheduler (order of addition doesn't affect final scheduling order)
        for request in requests:
            scheduler.add_request(request)

        # Execute scheduling
        output = scheduler.schedule()
        
        # Verify all requests are scheduled (resources are sufficient)
        assert len(output.scheduled_new_reqs) == 3

        # Verify scheduling order: longest wait first
        # Expected order: req1 (waited 9.0s), req2 (waited 8.0s), req0 (waited 7.0s)
        scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
        assert scheduled_req_ids == ["1", "2", "0"]
