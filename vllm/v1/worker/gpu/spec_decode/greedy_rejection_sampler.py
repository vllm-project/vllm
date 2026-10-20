# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared A33 kernels required by sharded greedy verification."""

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_target_argmax,
)


@triton.jit
def _greedy_global_argmax(
    maxima_ptr,
    indices_ptr,
    stats_stride,
    row,
    num_blocks,
    PADDED_BLOCKS: tl.constexpr,
    num_logits,
    TP_SIZE: tl.constexpr,
):
    if TP_SIZE == 1:
        return _compute_global_target_argmax(
            maxima_ptr,
            stats_stride,
            indices_ptr,
            stats_stride,
            row,
            num_blocks,
            PADDED_BLOCKS,
        )
    else:
        # Packed statistics are rank-major. Reconstruct each native vocabulary
        # block before selecting a block: a NaN in any shard invalidates it.
        ranks = tl.arange(0, TP_SIZE)
        blocks = tl.arange(0, PADDED_BLOCKS)
        offsets = (
            (ranks[:, None] * num_logits + row) * stats_stride + blocks[None, :]
        ) * 2
        values = tl.load(
            maxima_ptr + offsets,
            mask=blocks[None, :] < num_blocks,
            other=float("-inf"),
        )
        is_nan = values != values
        any_nan = tl.sum(is_nan.to(tl.int32), axis=0) > 0
        clean = tl.where(is_nan, float("-inf"), values)
        block_max = tl.max(clean, axis=0)
        source_rank = tl.argmax(clean, axis=0)
        scores = tl.where(any_nan, float("-inf"), block_max)
        winning_block = tl.argmax(scores, axis=0)
        indices = tl.load(
            maxima_ptr
            + ((source_rank * num_logits + row) * stats_stride + blocks) * 2
            + 1,
            mask=blocks < num_blocks,
            other=0,
        ).to(tl.int64)
        return tl.sum(tl.where(blocks == winning_block, indices, 0), axis=0)


@triton.jit(do_not_specialize=["num_logits"])
def _greedy_verify_and_count(
    maxima_ptr,
    indices_ptr,
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    seeds_ptr,
    rates_ptr,
    sampled_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    num_blocks,
    bonus_num_blocks,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    PADDED_BONUS_BLOCKS: tl.constexpr,
    SYNTHETIC: tl.constexpr,
    num_logits=0,
    TP_SIZE: tl.constexpr = 1,
):
    req = tl.program_id(0)
    state = tl.load(idx_mapping_ptr + req).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req + 1)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + state)
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for step in range(num_draft):
        if verifying:
            row = start + step
            token_index = tl.load(logits_indices_ptr + row + 1).to(tl.int64)
            draft = tl.load(input_ids_ptr + token_index).to(tl.int64)
            draft_position = tl.load(positions_ptr + token_index)
            local_position = tl.load(expanded_local_pos_ptr + row + 1)
            prefill_len = tl.load(prefill_len_ptr + state)
            placeholder = (local_position > 0) & (
                draft_position - local_position < prefill_len
            )
            valid_draft = (draft >= 0) & ~placeholder
            draft = tl.maximum(0, draft)
            target = _greedy_global_argmax(
                maxima_ptr,
                indices_ptr,
                bonus_num_blocks,
                row,
                num_blocks,
                PADDED_BLOCKS,
                num_logits,
                TP_SIZE,
            )
            if SYNTHETIC:
                pos_index = tl.load(logits_indices_ptr + row).to(tl.int64)
                pos = tl.load(positions_ptr + pos_index)
                uniform = tl_rand32(seed, pos, includes_zero=False)
                accepted = uniform < tl.load(rates_ptr + step)
            else:
                accepted = target == draft
            accepted &= valid_draft
            verifying = accepted
            accepted_length += accepted
            tl.store(
                sampled_ptr + req * OUTPUT_WIDTH + step,
                tl.where(accepted, draft, target),
            )
    if verifying:
        bonus = _greedy_global_argmax(
            maxima_ptr,
            indices_ptr,
            bonus_num_blocks,
            end - 1,
            bonus_num_blocks,
            PADDED_BONUS_BLOCKS,
            num_logits,
            TP_SIZE,
        )
        tl.store(sampled_ptr + req * OUTPUT_WIDTH + num_draft, bonus)

    prefilling = tl.load(seq_lens_ptr + req) < tl.load(prefill_len_ptr + state)
    sampled = tl.where(prefilling, 0, accepted_length + 1)
    rejected = tl.where(prefilling, 0, end - start - sampled)
    tl.store(num_sampled_ptr + req, sampled)
    tl.store(num_rejected_ptr + req, rejected)
