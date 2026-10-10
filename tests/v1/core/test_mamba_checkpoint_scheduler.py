# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Mamba prefix checkpointing in the V1 Scheduler.

Covers:
1. Same-step producer/consumer pairing: a consumer request arriving in the same
   scheduling step as its checkpoint producer inherits the producer's request ID,
   gets mamba_checkpoint_source_block_ids, and schedules alongside it.
2. Cross-step dependency state machine: when a checkpoint is not yet ready, consumer
   requests are skipped (waiting_for_mamba_checkpoint=True) without blocking unrelated
   requests, and are resumed once the checkpoint is marked ready.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256 as vllm_sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _create_hybrid_mamba_scheduler(
    num_blocks: int = 1000,
    block_size: int = BLOCK_SIZE,
    num_prefill_checkpoint_blocks: int = 0,
) -> Scheduler:
    from unittest.mock import patch
    from transformers import OPTConfig

    mock_cfg = OPTConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
    )
    mock_cfg.architectures = ["OPTForCausalLM"]

    with patch("vllm.config.model.get_config", return_value=mock_cfg):
        model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            seed=42,
            skip_tokenizer_init=True,
        )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            mamba_checkpoint_token="<|mamba_checkpoint|>",
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=0,
                    num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        log_stats=True,
    )


def test_scheduler_producer_checkpoint_stops_at_boundary():
    """Producer prefill chunk stops exactly at the mamba_checkpoint_position."""
    scheduler = _create_hybrid_mamba_scheduler()
    init_none_hash(vllm_sha256)
    block_hasher = get_request_block_hasher(BLOCK_SIZE, vllm_sha256)

    checkpoint_pos = 48
    tokens_producer = [10] * 100

    req_producer = Request(
        request_id="producer_0",
        prompt_token_ids=tokens_producer,
        sampling_params=SamplingParams(max_tokens=5),
        pooling_params=None,
        block_hasher=block_hasher,
        mamba_checkpoint_position=checkpoint_pos,
    )

    scheduler.add_request(req_producer)
    sched_out = scheduler.schedule()

    # Producer chunk stops at checkpoint boundary (48 tokens).
    assert [r.req_id for r in sched_out.scheduled_new_reqs] == ["producer_0"]
    assert sched_out.num_scheduled_tokens["producer_0"] == checkpoint_pos
    assert scheduler.kv_cache_manager.has_unready_checkpoint(req_producer)


def test_scheduler_cross_step_checkpoint_pending_and_wakeup():
    """Consumer is skipped while checkpoint is pending, and woken up once ready."""
    scheduler = _create_hybrid_mamba_scheduler()
    init_none_hash(vllm_sha256)
    block_hasher = get_request_block_hasher(BLOCK_SIZE, vllm_sha256)

    checkpoint_pos = 48
    tokens_producer = [10] * 100
    tokens_consumer = [10] * checkpoint_pos + [20] * 50

    req_producer = Request(
        request_id="producer_0",
        prompt_token_ids=tokens_producer,
        sampling_params=SamplingParams(max_tokens=5),
        pooling_params=None,
        block_hasher=block_hasher,
        mamba_checkpoint_position=checkpoint_pos,
    )
    req_consumer = Request(
        request_id="consumer_0",
        prompt_token_ids=tokens_consumer,
        sampling_params=SamplingParams(max_tokens=5),
        pooling_params=None,
        block_hasher=block_hasher,
        mamba_checkpoint_position=checkpoint_pos,
    )

    # Step 1: Producer is scheduled.
    scheduler.add_request(req_producer)
    out1 = scheduler.schedule()
    assert [r.req_id for r in out1.scheduled_new_reqs] == ["producer_0"]
    assert scheduler.kv_cache_manager.has_unready_checkpoint(req_producer)

    # Step 2: Consumer arrives while producer's checkpoint is still unready.
    scheduler.add_request(req_consumer)
    out2 = scheduler.schedule()
    assert len(out2.scheduled_new_reqs) == 0
    assert req_consumer.waiting_for_mamba_checkpoint is True

    # Step 3: Producer completes and marks checkpoint ready.
    scheduler.kv_cache_manager.mark_checkpoint_ready("producer_0")
    assert not scheduler.kv_cache_manager.has_unready_checkpoint(req_producer)

    # Step 4: Next schedule step resumes consumer, hitting the prefix cache.
    out3 = scheduler.schedule()
    assert [r.req_id for r in out3.scheduled_new_reqs] == ["consumer_0"]
    assert req_consumer.waiting_for_mamba_checkpoint is False
    assert req_consumer.mamba_prefix_producer_id is None
