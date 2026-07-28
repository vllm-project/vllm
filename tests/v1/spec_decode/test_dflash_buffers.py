# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.spec_decode.dflash import DFlashProposer

DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

NUM_SPECULATIVE_TOKENS = 3


def _dflash_vllm_config(max_num_seqs: int) -> VllmConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=8192,
        trust_remote_code=True,
    )
    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT_DIR,
        method="dflash",
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
    )
    return VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        ),
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
    )


@pytest.mark.parametrize("max_num_seqs", [33, 32])
def test_dflash_query_buffers_cover_cudagraph_padding(max_num_seqs: int):
    """The cudagraph dispatcher pads drafter batches up to the next capture
    size, which exceeds max_query_tokens whenever it is not itself a capture
    size (any odd max_num_seqs: e.g. 33 * 4 = 132 pads to 136 with the default
    ladder). The persistent query buffers must cover the padded size, or
    slicing them at that size silently yields short tensors and the drafter
    crashes during cudagraph warmup.
    """
    vllm_config = _dflash_vllm_config(max_num_seqs)
    proposer = DFlashProposer(vllm_config, torch.device("cuda"))

    max_query_tokens = max_num_seqs * (1 + NUM_SPECULATIVE_TOKENS)
    capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    padded_max_query = min(
        (size for size in capture_sizes if size >= max_query_tokens),
        default=max_query_tokens,
    )

    assert proposer.positions.shape[0] >= padded_max_query
    assert proposer._slot_mapping_buffer.shape[0] >= padded_max_query
    assert proposer.max_positions >= proposer.max_num_tokens + padded_max_query
