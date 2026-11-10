# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_SPECULATIVE_TOKENS = 3


def _dflash_speculative_config(
    num_speculative_tokens: int, **kwargs
) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    kwargs.setdefault("model", DFLASH_DRAFT_DIR)
    kwargs.setdefault("method", "dflash")
    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        num_speculative_tokens=num_speculative_tokens,
        **kwargs,
    )


def _create_dflash_scheduler(
    num_speculative_tokens: int, enable_prefix_caching: bool = False, **kwargs
) -> Scheduler:
    speculative_config = _dflash_speculative_config(num_speculative_tokens, **kwargs)
    model_config = speculative_config.target_model_config
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = None
    if speculative_config.prefill_context_kv_only:
        kv_transfer_config = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dflash_prefill_reserves_lookahead_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    assert scheduler.num_lookahead_tokens == NUM_SPECULATIVE_TOKENS + 1

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == BLOCK_SIZE
    # prefill block + one lookahead block
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2


@pytest.mark.parametrize(
    ("prefill_context_kv_only", "num_spec_tokens", "num_blocks"),
    [(True, 0, 1), (False, NUM_SPECULATIVE_TOKENS, 2)],
)
def test_dflash_prefill_context_kv_only_schedules_no_drafts(
    monkeypatch, prefill_context_kv_only, num_spec_tokens, num_blocks
):
    # The drafter only stores its context K/V (MRV2): no drafts, and no
    # lookahead block past the prompt's.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    scheduler = _create_dflash_scheduler(
        NUM_SPECULATIVE_TOKENS, prefill_context_kv_only=prefill_context_kv_only
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_spec_tokens_to_schedule == num_spec_tokens
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == num_blocks


def test_prefill_context_kv_only_full_prefix_hit_is_not_padded():
    # A full prefix-cache hit leaves one token to compute. With no drafts to
    # verify, it isn't padded to 1 + K rows.
    scheduler = _create_dflash_scheduler(
        NUM_SPECULATIVE_TOKENS,
        enable_prefix_caching=True,
        prefill_context_kv_only=True,
    )
    first, second = create_requests(
        num_requests=2,
        num_tokens=2 * BLOCK_SIZE + 1,
        block_size=BLOCK_SIZE,
        same_prompt=True,
    )
    scheduler.add_request(first)
    scheduler.schedule()
    scheduler.add_request(second)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[second.request_id] == 1
    assert not output.scheduled_spec_decode_tokens


def test_prefill_context_kv_only_config():
    config = _dflash_speculative_config(
        NUM_SPECULATIVE_TOKENS,
        prefill_context_kv_only=True,
        enable_adaptive_verification=True,
        num_speculative_tokens_per_batch_size=[(1, 16, 1)],
    )
    # Nothing is drafted, so the options for verifying drafts are turned off.
    assert not config.enable_adaptive_verification
    assert config.num_speculative_tokens_per_batch_size is None
    assert config.num_speculative_tokens == NUM_SPECULATIVE_TOKENS

    # Only an instance that sends KV has a decode instance to read it.
    with pytest.raises(ValueError, match="produces KV"):
        VllmConfig(model_config=config.target_model_config, speculative_config=config)

    with pytest.raises(ValueError, match="prefill_context_kv_only is not supported"):
        _dflash_speculative_config(
            NUM_SPECULATIVE_TOKENS,
            model=None,
            method="ngram",
            prefill_context_kv_only=True,
        )


def test_dflash_first_prefill_query_window_fits_allocated_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    query_positions = range(BLOCK_SIZE, BLOCK_SIZE + scheduler.num_lookahead_tokens)

    assert all(pos // BLOCK_SIZE < len(block_ids) for pos in query_positions)
