# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused generated-token top-logprobs helper.

This optional CUDA path computes the generated-token logprob, rank, and top-N
logprobs directly from logits. It avoids materializing a full
``log_softmax(logits)`` tensor for the common sampled-token logprobs path.
"""

from __future__ import annotations

import os

import torch

from vllm.v1.outputs import LogprobsTensors

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on optional GPU runtime.
    triton = None
    tl = None

_ENV_VAR = "VLLM_ENABLE_FUSED_TOP_LOGPROBS"


def fused_top_logprobs_enabled() -> bool:
    return os.environ.get(_ENV_VAR, "0").strip().lower() in ("1", "true")


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _launch_config(
    vocab_size: int,
    top_n: int,
    batch_size: int,
) -> tuple[int, int, int, int]:
    if top_n <= 0 or top_n > 64:
        raise ValueError("top_n must be in [1, 64]")
    if vocab_size <= 0 or vocab_size > 262_144:
        raise ValueError("vocab_size must be in [1, 262144]")
    if batch_size <= 0 or batch_size > 64:
        raise ValueError("batch_size must be in [1, 64]")

    block_vocab = 2048 if batch_size <= 4 else 1024
    blocks_per_row = (vocab_size + block_vocab - 1) // block_vocab
    num_warps = 8 if block_vocab == 2048 else 4
    return block_vocab, blocks_per_row, num_warps, 1


def fused_top_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    num_logprobs: int,
) -> LogprobsTensors | None:
    """Return vLLM-compatible top-logprobs, or ``None`` on unsupported input."""

    if triton is None:
        return None
    if logits.ndim != 2 or token_ids.ndim != 1:
        return None
    if not logits.is_cuda or not token_ids.is_cuda:
        return None
    if token_ids.dtype != torch.int64:
        return None

    batch_size, vocab_size = logits.shape
    if token_ids.shape != (batch_size,):
        return None
    try:
        block_vocab, blocks_per_row, num_warps, num_stages = _launch_config(
            int(vocab_size),
            int(num_logprobs),
            int(batch_size),
        )
    except ValueError:
        return None

    output_token_ids = torch.empty(
        (batch_size, num_logprobs + 1),
        device=logits.device,
        dtype=torch.int32,
    )
    output_logprobs = torch.empty(
        (batch_size, num_logprobs + 1),
        device=logits.device,
        dtype=torch.float32,
    )
    output_ranks = torch.empty(
        (batch_size,),
        device=logits.device,
        dtype=torch.int32,
    )
    partial_values = torch.empty(
        (batch_size, blocks_per_row, num_logprobs),
        device=logits.device,
        dtype=torch.float32,
    )
    partial_tokens = torch.empty(
        (batch_size, blocks_per_row, num_logprobs),
        device=logits.device,
        dtype=torch.int64,
    )
    partial_max = torch.empty(
        (batch_size, blocks_per_row),
        device=logits.device,
        dtype=torch.float32,
    )
    partial_sum_exp = torch.empty_like(partial_max)
    partial_ranks = torch.empty(
        (batch_size, blocks_per_row),
        device=logits.device,
        dtype=torch.int32,
    )

    _top_logprobs_partial_kernel[(batch_size, blocks_per_row)](
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        VOCAB=int(vocab_size),
        TOP_N=int(num_logprobs),
        BLOCK_VOCAB=block_vocab,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _top_logprobs_reduce_kernel[(batch_size,)](
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        output_token_ids,
        output_logprobs,
        output_ranks,
        VOCAB=int(vocab_size),
        TOP_N=int(num_logprobs),
        BLOCKS_PER_ROW=blocks_per_row,
        REDUCE_BLOCK=_next_power_of_2(blocks_per_row),
        CANDIDATE_BLOCK=_next_power_of_2(blocks_per_row * int(num_logprobs)),
        num_warps=8,
        num_stages=1,
    )
    return LogprobsTensors(output_token_ids, output_logprobs, output_ranks)


if triton is not None:  # pragma: no cover - requires CUDA.

    @triton.jit
    def _top_logprobs_partial_kernel(
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        VOCAB: tl.constexpr,
        TOP_N: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(
            logits + row * VOCAB + offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(values, axis=0)
        exp_values = tl.exp(values - block_max)
        block_sum = tl.sum(tl.where(mask, exp_values, 0.0), axis=0)
        sampled_token = tl.load(token_ids + row).to(tl.int64)
        sampled_value = tl.load(logits + row * VOCAB + sampled_token).to(tl.float32)
        # Match batched_count_greater_than, which uses >= for rank.
        rank_count = tl.sum(tl.where((values >= sampled_value) & mask, 1, 0), axis=0)

        block_offset = row * tl.cdiv(VOCAB, BLOCK_VOCAB) + block
        tl.store(partial_max + block_offset, block_max)
        tl.store(partial_sum_exp + block_offset, block_sum)
        tl.store(partial_ranks + block_offset, rank_count)

        base = block_offset * TOP_N
        for rank in tl.static_range(0, TOP_N):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & mask
            token_values = tl.where(is_max, offsets, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(partial_values + base + rank, max_value)
            tl.store(partial_tokens + base + rank, token)
            values = tl.where(offsets == token, -float("inf"), values)

    @triton.jit
    def _top_logprobs_reduce_kernel(
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        output_token_ids,
        output_logprobs,
        output_ranks,
        VOCAB: tl.constexpr,
        TOP_N: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        REDUCE_BLOCK: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_offsets = tl.arange(0, REDUCE_BLOCK)
        block_mask = block_offsets < BLOCKS_PER_ROW
        block_max = tl.load(
            partial_max + row * BLOCKS_PER_ROW + block_offsets,
            mask=block_mask,
            other=-float("inf"),
        ).to(tl.float32)
        block_sum = tl.load(
            partial_sum_exp + row * BLOCKS_PER_ROW + block_offsets,
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)
        global_max = tl.max(block_max, axis=0)
        total_exp = tl.sum(
            tl.where(block_mask, block_sum * tl.exp(block_max - global_max), 0.0),
            axis=0,
        )
        log_denom = tl.log(total_exp) + global_max

        rank_counts = tl.load(
            partial_ranks + row * BLOCKS_PER_ROW + block_offsets,
            mask=block_mask,
            other=0,
        )
        sampled_rank = tl.sum(rank_counts, axis=0)
        sampled_token = tl.load(token_ids + row).to(tl.int64)
        sampled_value = tl.load(logits + row * VOCAB + sampled_token).to(tl.float32)

        out_base = row * (TOP_N + 1)
        tl.store(output_token_ids + out_base, sampled_token)
        tl.store(output_logprobs + out_base, sampled_value - log_denom)
        tl.store(output_ranks + row, sampled_rank)

        candidate_offsets = tl.arange(0, CANDIDATE_BLOCK)
        candidate_count = BLOCKS_PER_ROW * TOP_N
        candidate_mask = candidate_offsets < candidate_count
        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )
        for rank in tl.static_range(0, TOP_N):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(output_token_ids + out_base + rank + 1, token)
            tl.store(output_logprobs + out_base + rank + 1, max_value - log_denom)
            values = tl.where(tokens == token, -float("inf"), values)
