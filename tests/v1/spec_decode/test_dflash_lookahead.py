# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
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
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Matches defaults from tests/v1/spec_decode/test_eagle.py
DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_SPECULATIVE_TOKENS = 3


def _dflash_speculative_config(num_speculative_tokens: int) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT_DIR,
        method="dflash",
        num_speculative_tokens=num_speculative_tokens,
    )


def _create_dflash_scheduler(num_speculative_tokens: int) -> Scheduler:
    speculative_config = _dflash_speculative_config(num_speculative_tokens)
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
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
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


def test_dflash_drafter_window_reserves_bonus_token():
    # DFlash's drafter window is num_spec + 1 (the extra slot is the bonus token),
    # so max_seq_len + num_spec + 1 must stay within the draft model's max len.
    input_fits_in_drafter = GPUModelRunner._input_fits_in_drafter
    dflash_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=_dflash_speculative_config(NUM_SPECULATIVE_TOKENS),
    )
    # window = 4, so 96 fits (96 + 4 == 100) but 97 does not (97 + 4 == 101)
    assert input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=96))
    assert not input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=97))
    assert not input_fits_in_drafter(dflash_runner, None)  # no metadata

    # Other drafters don't reserve the bonus token, so 97 fits (97 + 3 == 100).
    plain_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
    )
    assert input_fits_in_drafter(plain_runner, SimpleNamespace(max_seq_len=97))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("sliding_window", "expected_context_slots"),
    [
        (0, [40, 41, 42, 43, 44, 45]),
        (3, [-1, -1, -1, -1, 44, 45]),
    ],
)
def test_dflash_input_copy_filters_context_by_sliding_window(
    sliding_window: int,
    expected_context_slots: list[int],
):
    device = "cuda"
    num_reqs = 1
    block_size = 4
    num_speculative_tokens = 2
    num_query_per_req = num_speculative_tokens + 1
    total_input_tokens = 6
    kernel_block_size = 16

    next_token_ids = torch.tensor([99], dtype=torch.int64, device=device)
    target_positions = torch.arange(
        total_input_tokens, dtype=torch.int64, device=device
    )
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor(
        [0, total_input_tokens], dtype=torch.int32, device=device
    )
    empty_rejected_tokens = torch.empty(0, dtype=torch.int32, device=device)

    out_input_ids = torch.full(
        (num_reqs * num_query_per_req,), -1, dtype=torch.int64, device=device
    )
    out_context_positions = torch.full(
        (total_input_tokens,), -1, dtype=torch.int64, device=device
    )
    out_query_positions = torch.full(
        (num_reqs * num_query_per_req,), -1, dtype=torch.int64, device=device
    )
    out_context_slot_mapping = torch.full(
        (total_input_tokens,), -9, dtype=torch.int64, device=device
    )
    out_query_slot_mapping = torch.full(
        (num_reqs * num_query_per_req,), -9, dtype=torch.int64, device=device
    )
    out_token_indices = torch.full(
        (num_reqs * num_speculative_tokens,), -1, dtype=torch.int64, device=device
    )

    copy_and_expand_dflash_inputs_kernel[(num_reqs, 1)](
        next_token_ids,
        target_positions,
        out_input_ids,
        out_context_positions,
        out_query_positions,
        out_context_slot_mapping,
        out_query_slot_mapping,
        out_token_indices,
        block_table,
        block_table.stride(0),
        query_start_loc,
        empty_rejected_tokens,
        32000,
        block_size,
        sliding_window,
        num_query_per_req,
        num_speculative_tokens,
        total_input_tokens,
        BLOCK_SIZE=kernel_block_size,
        HAS_NUM_REJECTED=False,
    )
    torch.cuda.synchronize()

    assert out_context_positions.cpu().tolist() == list(range(total_input_tokens))
    assert out_query_positions.cpu().tolist() == [6, 7, 8]
    assert out_input_ids.cpu().tolist() == [99, 32000, 32000]
    assert out_token_indices.cpu().tolist() == [1, 2]
    assert out_context_slot_mapping.cpu().tolist() == expected_context_slots
    assert out_query_slot_mapping.cpu().tolist() == [46, 47, 48]
