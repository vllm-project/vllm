# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 16
NUM_BLOCKS = 6
MAX_MODEL_LEN = 1000
LONG_PREFILL_THRESHOLD = 15
EOS_TOKEN_ID = 50256


def build_scheduler():
    model_config = ModelConfig(
        model="facebook/opt-350m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=5,
        max_num_batched_tokens=200,
        max_model_len=MAX_MODEL_LEN,
        long_prefill_token_threshold=LONG_PREFILL_THRESHOLD,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="priority",
        watermark=0.0,
        scheduler_reserve_full_isl=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


_hash_init = False


def make_request(req_id, num_tokens, priority, arrival_time):
    global _hash_init
    if not _hash_init:
        init_none_hash(sha256)
        _hash_init = True
    block_hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    sp = SamplingParams(ignore_eos=True, max_tokens=50)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=[7] * num_tokens,
        sampling_params=sp,
        pooling_params=None,
        priority=priority,
        arrival_time=arrival_time,
        block_hasher=block_hasher,
    )


def mock_output(out, tok=100):
    ids = list(out.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[[tok] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_priority_scheduler_preempt_skipped_request():
    scheduler = build_scheduler()

    # R1: Add request A (Worst priority)
    A = make_request("A", num_tokens=200, priority=9, arrival_time=1.0)
    scheduler.add_request(A)
    out = scheduler.schedule()
    scheduler.update_from_output(out, mock_output(out))

    # R2: Add request B (Best priority)
    B = make_request("B", num_tokens=15, priority=0, arrival_time=2.0)
    scheduler.add_request(B)
    out = scheduler.schedule()
    scheduler.update_from_output(out, mock_output(out))

    # R3: Add request C (Middle priority)
    C = make_request("C", num_tokens=1, priority=1, arrival_time=3.0)
    scheduler.add_request(C)
    out = scheduler.schedule()
    scheduler.update_from_output(out, mock_output(out))

    # R4: Trigger preemption
    out = scheduler.schedule(throttle_prefills=True)

    is_c_scheduled = "C" in out.num_scheduled_tokens

    assert is_c_scheduled, (
        "Bug present: Request C was silently skipped during scheduling "
        "because the req_index was not decremented after preempting A."
    )
    assert C.status == RequestStatus.RUNNING
