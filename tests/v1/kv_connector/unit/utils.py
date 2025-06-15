# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

from vllm import SamplingParams
from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig,
                         ModelConfig, SchedulerConfig, VllmConfig)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0
    assert len(scheduler.finished_recving_kv_req_ids) == 0
    assert len(scheduler._cached_reqs_data) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].
               req_to_blocks) == 0
    assert len(scheduler.kv_cache_manager.req_to_block_hashes) == 0
    assert len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].
               num_cached_block) == 0
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)
    assert num_free_blocks == (
        scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    # NOTE(rob): just the ref count on blocks will be 0. The hash
    # value, etc will remain since we lazily evict for prefix cache.
    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0


def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
) -> VllmConfig:
    """Initialize VllmConfig For Testing."""
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
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    return VllmConfig(scheduler_config=scheduler_config,
                      model_config=model_config,
                      cache_config=cache_config,
                      kv_transfer_config=kv_transfer_config,
                      device_config=DeviceConfig("cpu"))


def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_request(
    request_id: int,
    num_tokens: int = 10,
    max_tokens: int = 16,
    do_remote_decode: bool = False,
    do_remote_prefill: bool = False,
    use_all_1s_for_prompt_tokens: bool = False,
    num_remote_blocks: int = 3,
) -> Request:
    """Make dummy request for testing."""

    kv_transfer_params: Optional[dict[str, Any]] = None

    if do_remote_decode:
        assert not do_remote_prefill
        kv_transfer_params = dict(do_remote_prefill=False,
                                  do_remote_decode=True)
    elif do_remote_prefill:
        kv_transfer_params = dict(do_remote_prefill=True,
                                  do_remote_decode=False,
                                  remote_engine_id="my-engine-id",
                                  remote_block_ids=list(
                                      range(num_remote_blocks)),
                                  remote_host="my-host",
                                  remote_port=1234)

    max_tokens = 1 if do_remote_decode else max_tokens
    sampling_params = SamplingParams(max_tokens=max_tokens)

    if use_all_1s_for_prompt_tokens:
        prompt_token_ids = [1] * num_tokens
    else:
        prompt_token_ids = [i * request_id for i in range(num_tokens)]

    req = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        multi_modal_inputs=None,
        multi_modal_placeholders=None,
        multi_modal_hashes=None,
        eos_token_id=EOS_TOKEN_ID,
    )
    req.kv_transfer_params = kv_transfer_params
    return req


def create_model_runner_output(
    reqs: list[Request],
    finished_sending: Optional[list[str]] = None,
    finished_recving: Optional[list[str]] = None,
    use_eos: bool = False,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""

    # Make request data.
    req_ids = [req.request_id for req in reqs]
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    # Make sampled tokens.
    sampled_token = EOS_TOKEN_ID if use_eos else 0
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    # Make output data structure.
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        finished_sending=finished_sending,
        finished_recving=finished_recving,
    )
