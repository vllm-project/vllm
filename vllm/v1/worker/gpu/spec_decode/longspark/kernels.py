# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from LongSpark at commit 9ab6526e481bbea67489f3a73ebc782218d56f71:
# https://github.com/Hao-Yuan-He/LongSpark/tree/9ab6526e481bbea67489f3a73ebc782218d56f71/vendor/sglang/python/sglang/srt/speculative
#
# MIT License
# Copyright (c) 2026 The LongSpark Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Shape-parameterized Triton kernels used by LongSpark speculative decoding."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["mask_token_id"])
def _prepare_shifted_input_kernel(
    token_ids,
    latest,
    embedding,
    position,
    output,
    mask_token_id,
    token_stride: tl.constexpr,
    latest_stride: tl.constexpr,
    embedding_stride: tl.constexpr,
    position_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    output_step_stride: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1)
    dimensions = tl.program_id(2) * HIDDEN_BLOCK + tl.arange(0, HIDDEN_BLOCK)
    valid = dimensions < HIDDEN_SIZE
    anchor = tl.load(token_ids + batch * token_stride)
    token = tl.where(step == 0, anchor, mask_token_id)
    token_state = tl.load(
        embedding + token * embedding_stride + dimensions,
        mask=valid,
        other=0.0,
    )
    position_state = tl.load(
        position + step * position_stride + dimensions,
        mask=valid,
        other=0.0,
    )
    output_row = output + batch * output_batch_stride + step * output_step_stride
    tl.store(output_row + dimensions, token_state + position_state, mask=valid)
    latest_state = tl.load(
        latest + batch * latest_stride + dimensions,
        mask=valid,
        other=0.0,
    )
    tl.store(output_row + HIDDEN_SIZE + dimensions, latest_state, mask=valid)


def prepare_longspark_shifted_input(
    token_ids: torch.Tensor,
    latest: torch.Tensor,
    embedding_weight: torch.Tensor,
    position_weight: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Build LongSpark's shifted token/latest input in one exact BF16 pass."""
    if token_ids.ndim != 1 or latest.ndim != 2:
        raise ValueError("LongSpark shifted input requires [batch] tokens and states")
    batch_size, hidden_size = latest.shape
    block_size = position_weight.shape[0] if position_weight.ndim == 2 else 0
    if (
        batch_size <= 0
        or hidden_size <= 0
        or block_size <= 0
        or token_ids.shape[0] != batch_size
        or embedding_weight.ndim != 2
        or embedding_weight.shape[1] != hidden_size
        or position_weight.shape != (block_size, hidden_size)
        or mask_token_id < 0
        or mask_token_id >= embedding_weight.shape[0]
        or not token_ids.is_cuda
        or token_ids.dtype not in (torch.int32, torch.int64)
        or latest.dtype != torch.bfloat16
        or embedding_weight.dtype != torch.bfloat16
        or position_weight.dtype != torch.bfloat16
        or any(
            tensor.device != latest.device
            for tensor in (token_ids, embedding_weight, position_weight)
        )
        or latest.stride(1) != 1
        or embedding_weight.stride(1) != 1
        or position_weight.stride(1) != 1
    ):
        raise ValueError("LongSpark shifted input tensors violate the kernel contract")
    output = torch.empty(
        (batch_size, block_size, 2 * hidden_size),
        dtype=latest.dtype,
        device=latest.device,
    )
    hidden_block = 256
    _prepare_shifted_input_kernel[
        (batch_size, block_size, triton.cdiv(hidden_size, hidden_block))
    ](
        token_ids,
        latest,
        embedding_weight,
        position_weight,
        output,
        mask_token_id,
        token_ids.stride(0),
        latest.stride(0),
        embedding_weight.stride(0),
        position_weight.stride(0),
        output.stride(0),
        output.stride(1),
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        HIDDEN_BLOCK=hidden_block,
        num_warps=4,
    )
    return output


@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _markov_project_argmax_stage1(
    base,
    token_ids,
    embedding_weight,
    weight,
    partial_values,
    partial_ids,
    vocab_size,
    base_stride_b: tl.constexpr,
    token_stride_b: tl.constexpr,
    embedding_weight_stride_v: tl.constexpr,
    weight_stride_v: tl.constexpr,
    partial_stride_b: tl.constexpr,
    chunks: tl.constexpr,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    MARKOV_RANK: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    REVERSE: tl.constexpr,
) -> None:
    batch_group = tl.program_id(0)
    chunk_index = tl.program_id(1)
    if REVERSE:
        chunk_index = chunks - 1 - chunk_index
    batch_offsets = batch_group * BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    rank_offsets = tl.arange(0, RANK_BLOCK)
    vocab_offsets = chunk_index * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    valid_batch = batch_offsets < BATCH_SIZE
    valid_rank = rank_offsets < MARKOV_RANK
    valid_vocab = vocab_offsets < vocab_size

    tokens = tl.load(
        token_ids + batch_offsets * token_stride_b,
        mask=valid_batch,
        other=0,
    ).to(tl.int64)
    embeddings = tl.load(
        embedding_weight
        + tokens[:, None] * embedding_weight_stride_v
        + rank_offsets[None, :],
        mask=valid_batch[:, None] & valid_rank[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    base_values = tl.load(
        base + batch_offsets[:, None] * base_stride_b + vocab_offsets[None, :],
        mask=valid_batch[:, None] & valid_vocab[None, :],
        other=-float("inf"),
        eviction_policy="evict_first",
    ).to(tl.float32)
    weights = tl.load(
        weight + rank_offsets[:, None] + vocab_offsets[None, :] * weight_stride_v,
        mask=valid_rank[:, None] & valid_vocab[None, :],
        other=0.0,
        cache_modifier=".cg",
    )
    bias = tl.dot(embeddings, weights).to(tl.bfloat16).to(tl.float32)
    scores = base_values + bias
    best_values = tl.max(scores, axis=1)
    best_ids = tl.min(
        tl.where(scores == best_values[:, None], vocab_offsets[None, :], vocab_size),
        axis=1,
    )
    output_offsets = batch_offsets * partial_stride_b + chunk_index
    tl.store(partial_values + output_offsets, best_values, mask=valid_batch)
    tl.store(partial_ids + output_offsets, best_ids, mask=valid_batch)


@triton.jit
def _add_argmax_stage1(
    base,
    bias,
    partial_values,
    partial_ids,
    vocab_size,
    base_stride_b: tl.constexpr,
    bias_stride_b: tl.constexpr,
    partial_stride_b: tl.constexpr,
    chunks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    batch_index = tl.program_id(0)
    chunk_index = tl.program_id(1)
    offsets = chunk_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < vocab_size
    base_values = tl.load(
        base + batch_index * base_stride_b + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    bias_values = tl.load(
        bias + batch_index * bias_stride_b + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    scores = base_values + bias_values
    best_value = tl.max(scores, axis=0)
    best_id = tl.min(tl.where(scores == best_value, offsets, vocab_size), axis=0)
    output_offset = batch_index * partial_stride_b + chunk_index
    tl.store(partial_values + output_offset, best_value)
    tl.store(partial_ids + output_offset, best_id)


@triton.jit
def _add_argmax_stage2(
    partial_values,
    partial_ids,
    output,
    vocab_size,
    output_stride: tl.constexpr,
    partial_stride_b: tl.constexpr,
    chunks: tl.constexpr,
    REDUCE_BLOCK_SIZE: tl.constexpr,
) -> None:
    batch_index = tl.program_id(0)
    chunk_offsets = tl.arange(0, REDUCE_BLOCK_SIZE)
    valid = chunk_offsets < chunks
    input_offsets = batch_index * partial_stride_b + chunk_offsets
    values = tl.load(
        partial_values + input_offsets,
        mask=valid,
        other=-float("inf"),
    )
    token_ids = tl.load(
        partial_ids + input_offsets,
        mask=valid,
        other=vocab_size,
    )
    best_value = tl.max(values, axis=0)
    best_id = tl.min(tl.where(values == best_value, token_ids, vocab_size), axis=0)
    tl.store(output + batch_index * output_stride, best_id)


@triton.jit
def _markov_partial_topk_tiles(
    partial_values,
    partial_ids,
    tile_values,
    tile_ids,
    partial_stride_b: tl.constexpr,
    tile_stride_b: tl.constexpr,
    tile_stride_g: tl.constexpr,
    chunks: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
) -> None:
    batch_index = tl.program_id(0)
    tile_index = tl.program_id(1)
    offsets = tile_index * TILE_SIZE + tl.arange(0, TILE_SIZE)
    valid = offsets < chunks
    values = tl.load(
        partial_values + batch_index * partial_stride_b + offsets,
        mask=valid,
        other=-float("inf"),
    )
    token_ids = tl.load(
        partial_ids + batch_index * partial_stride_b + offsets,
        mask=valid,
        other=2147483647,
    )
    top_values = tl.topk(values, TOP_K)
    matches = tl.expand_dims(values, 1) == tl.expand_dims(top_values, 0)
    top_ids = tl.min(
        tl.where(matches, tl.expand_dims(token_ids, 1), 2147483647), axis=0
    )
    output_offsets = (
        batch_index * tile_stride_b + tile_index * tile_stride_g + tl.arange(0, TOP_K)
    )
    tl.store(tile_values + output_offsets, top_values)
    tl.store(tile_ids + output_offsets, top_ids)


@triton.jit
def _markov_tile_topk(
    tile_values,
    tile_ids,
    candidate_ids,
    tile_stride_b: tl.constexpr,
    candidate_stride_b: tl.constexpr,
    candidate_count: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
) -> None:
    batch_index = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_SIZE)
    valid = offsets < candidate_count
    values = tl.load(
        tile_values + batch_index * tile_stride_b + offsets,
        mask=valid,
        other=-float("inf"),
    )
    token_ids = tl.load(
        tile_ids + batch_index * tile_stride_b + offsets,
        mask=valid,
        other=2147483647,
    )
    top_values = tl.topk(values, TOP_K)
    matches = tl.expand_dims(values, 1) == tl.expand_dims(top_values, 0)
    top_ids = tl.min(
        tl.where(matches, tl.expand_dims(token_ids, 1), 2147483647), axis=0
    )
    tl.store(
        candidate_ids + batch_index * candidate_stride_b + tl.arange(0, TOP_K),
        top_ids,
    )


@triton.jit
def _markov_refine_candidates(
    base,
    token_ids,
    embedding_weight,
    weight,
    candidate_ids,
    output,
    base_stride_b: tl.constexpr,
    token_stride_b: tl.constexpr,
    embedding_stride_v: tl.constexpr,
    weight_stride_v: tl.constexpr,
    candidate_stride_b: tl.constexpr,
    output_stride: tl.constexpr,
    MARKOV_RANK: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
) -> None:
    batch_index = tl.program_id(0)
    rank_offsets = tl.arange(0, RANK_BLOCK)
    candidate_offsets = tl.arange(0, TOP_K)
    candidates = tl.load(
        candidate_ids + batch_index * candidate_stride_b + candidate_offsets
    )
    previous = tl.load(token_ids + batch_index * token_stride_b).to(tl.int64)
    embedding = tl.load(
        embedding_weight + previous * embedding_stride_v + rank_offsets,
        mask=rank_offsets < MARKOV_RANK,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        weight + candidates[:, None] * weight_stride_v + rank_offsets[None, :],
        mask=rank_offsets[None, :] < MARKOV_RANK,
        other=0.0,
    ).to(tl.float32)
    bias = tl.sum(weights * embedding[None, :], axis=1)
    bias = bias.to(tl.bfloat16).to(tl.float32)
    scores = (
        tl.load(base + batch_index * base_stride_b + candidates).to(tl.float32) + bias
    )
    best_value = tl.max(scores, axis=0)
    best_id = tl.min(tl.where(scores == best_value, candidates, 2147483647), axis=0)
    tl.store(output + batch_index * output_stride, best_id)


def fused_add_argmax(
    base: torch.Tensor,
    bias: torch.Tensor,
    *,
    partial_values: torch.Tensor,
    partial_ids: torch.Tensor,
    output: torch.Tensor,
    block_size: int = 8192,
) -> torch.Tensor:
    """Return ``argmax(base + bias)`` without materializing the sum."""
    if base.ndim != 2 or bias.shape != base.shape:
        raise ValueError("base and bias must have the same [batch, vocab] shape")
    if not base.is_cuda or not bias.is_cuda or base.device != bias.device:
        raise ValueError("fused LongSpark Markov argmax requires one CUDA device")
    if base.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported base dtype: {base.dtype}")
    if bias.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported bias dtype: {bias.dtype}")
    if base.stride(1) != 1 or bias.stride(1) != 1:
        raise ValueError("base and bias vocabulary rows must be contiguous")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    batch_size, vocab_size = base.shape
    chunks = triton.cdiv(vocab_size, block_size)
    if partial_values.shape[0] < batch_size or partial_values.shape[1] != chunks:
        raise ValueError("partial-value workspace has an invalid shape")
    if partial_ids.shape[0] < batch_size or partial_ids.shape[1] != chunks:
        raise ValueError("partial-id workspace has an invalid shape")
    if (
        partial_values.dtype != torch.float32
        or partial_ids.dtype != torch.int32
        or not partial_values.is_contiguous()
        or not partial_ids.is_contiguous()
        or partial_values.device != base.device
        or partial_ids.device != base.device
    ):
        raise ValueError("partial workspaces have an invalid dtype, layout, or device")
    if (
        output.numel() < batch_size
        or output.dtype != torch.int64
        or output.device != base.device
    ):
        raise ValueError("output must provide one int64 element per request")
    _add_argmax_stage1[(batch_size, chunks)](
        base,
        bias,
        partial_values,
        partial_ids,
        vocab_size,
        base.stride(0),
        bias.stride(0),
        partial_values.stride(0),
        chunks=chunks,
        BLOCK_SIZE=block_size,
    )
    _add_argmax_stage2[(batch_size,)](
        partial_values,
        partial_ids,
        output,
        vocab_size,
        output.stride(0),
        partial_values.stride(0),
        chunks=chunks,
        REDUCE_BLOCK_SIZE=triton.next_power_of_2(chunks),
    )
    return output[:batch_size]


def fused_markov_argmax(
    base: torch.Tensor,
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    weight: torch.Tensor,
    *,
    reverse: bool = False,
    partial_values: torch.Tensor,
    partial_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fuse Markov embedding, rank-R projection, base add, and argmax."""
    if base.ndim != 2:
        raise ValueError("base must have shape [batch, vocab]")
    batch_size, vocab_size = base.shape
    if batch_size <= 0:
        raise ValueError("fused Markov projection requires a non-empty batch")
    if token_ids.ndim != 1 or token_ids.shape[0] != batch_size:
        raise ValueError("token_ids must have shape [batch]")
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, rank]")
    markov_rank = embedding_weight.shape[1]
    if markov_rank <= 0 or weight.shape != (vocab_size, markov_rank):
        raise ValueError("weight must have shape [vocab, rank]")
    if (
        not base.is_cuda
        or token_ids.device != base.device
        or embedding_weight.device != base.device
        or weight.device != base.device
        or base.dtype != torch.bfloat16
        or token_ids.dtype not in (torch.int32, torch.int64)
        or embedding_weight.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or base.stride(1) != 1
        or embedding_weight.stride(1) != 1
        or weight.stride(1) != 1
    ):
        raise ValueError(
            "fused Markov projection requires integer CUDA tokens and "
            "contiguous BF16 CUDA rows"
        )

    batch_block = min(16, max(2, triton.next_power_of_2(batch_size)))
    block_vocab = 64 if batch_block >= 8 else 32
    chunks = triton.cdiv(vocab_size, block_vocab)
    if (
        partial_values.shape[0] < batch_size
        or partial_values.shape[1] < chunks
        or partial_values.dtype != torch.float32
        or not partial_values.is_contiguous()
        or partial_values.device != base.device
        or partial_ids.shape[0] < batch_size
        or partial_ids.shape[1] < chunks
        or partial_ids.dtype != torch.int32
        or not partial_ids.is_contiguous()
        or partial_ids.device != base.device
    ):
        raise ValueError("partial workspaces have an invalid shape, dtype, or device")
    if (
        output.numel() < batch_size
        or output.dtype != torch.int64
        or output.device != base.device
    ):
        raise ValueError("output must provide one int64 element per request")

    rank_block = max(16, triton.next_power_of_2(markov_rank))
    _markov_project_argmax_stage1[(triton.cdiv(batch_size, batch_block), chunks)](
        base,
        token_ids,
        embedding_weight,
        weight,
        partial_values,
        partial_ids,
        vocab_size,
        base.stride(0),
        token_ids.stride(0),
        embedding_weight.stride(0),
        weight.stride(0),
        partial_values.stride(0),
        chunks=chunks,
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=batch_block,
        MARKOV_RANK=markov_rank,
        RANK_BLOCK=rank_block,
        BLOCK_VOCAB=block_vocab,
        REVERSE=reverse,
        num_warps=4,
        num_stages=1,
    )
    _add_argmax_stage2[(batch_size,)](
        partial_values,
        partial_ids,
        output,
        vocab_size,
        output.stride(0),
        partial_values.stride(0),
        chunks=chunks,
        REDUCE_BLOCK_SIZE=triton.next_power_of_2(chunks),
    )
    return output[:batch_size]


def fused_markov_refined_argmax(
    base: torch.Tensor,
    token_ids: torch.Tensor,
    candidate_embedding_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
    exact_embedding_weight: torch.Tensor,
    exact_weight: torch.Tensor,
    *,
    reverse: bool = False,
    partial_values: torch.Tensor,
    partial_ids: torch.Tensor,
    tile_values: torch.Tensor,
    tile_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    output: torch.Tensor,
    top_k: int = 32,
) -> torch.Tensor:
    """Retrieve low-rank candidates and re-score them with exact BF16 weights."""
    if base.ndim != 2:
        raise ValueError("base must have shape [batch, vocab]")
    batch_size, vocab_size = base.shape
    exact_rank = exact_embedding_weight.shape[1]
    candidate_rank = candidate_embedding_weight.shape[1]
    if (
        batch_size <= 0
        or token_ids.shape != (batch_size,)
        or candidate_rank <= 0
        or candidate_rank >= exact_rank
        or candidate_weight.shape != (vocab_size, candidate_rank)
        or exact_embedding_weight.shape != (vocab_size, exact_rank)
        or exact_weight.shape != (vocab_size, exact_rank)
        or top_k <= 0
        or top_k & (top_k - 1)
    ):
        raise ValueError("invalid refined Markov projection shapes")
    tensors = (
        base,
        token_ids,
        candidate_embedding_weight,
        candidate_weight,
        exact_embedding_weight,
        exact_weight,
    )
    if (
        any(tensor.device != base.device for tensor in tensors)
        or not base.is_cuda
        or base.dtype != torch.bfloat16
        or token_ids.dtype not in (torch.int32, torch.int64)
        or any(tensor.dtype != torch.bfloat16 for tensor in tensors[2:])
        or any(tensor.stride(1) != 1 for tensor in tensors[2:])
    ):
        raise ValueError("refined Markov projection requires contiguous CUDA BF16")

    batch_block = min(16, max(2, triton.next_power_of_2(batch_size)))
    block_vocab = 64 if batch_block >= 8 else 32
    chunks = triton.cdiv(vocab_size, block_vocab)
    tile_size = 512
    num_tiles = triton.cdiv(chunks, tile_size)
    if (
        partial_values.shape[0] < batch_size
        or partial_values.shape[1] < chunks
        or partial_values.dtype != torch.float32
        or partial_ids.shape[0] < batch_size
        or partial_ids.shape[1] < chunks
        or partial_ids.dtype != torch.int32
        or tile_values.shape[0] < batch_size
        or tile_values.shape[1] < num_tiles
        or tile_values.shape[2] < top_k
        or tile_values.dtype != torch.float32
        or tile_ids.shape[0] < batch_size
        or tile_ids.shape[1] < num_tiles
        or tile_ids.shape[2] < top_k
        or tile_ids.dtype != torch.int32
        or candidate_ids.shape[0] < batch_size
        or candidate_ids.shape[1] < top_k
        or candidate_ids.dtype != torch.int32
        or output.numel() < batch_size
        or output.dtype != torch.int64
        or any(
            tensor.device != base.device
            for tensor in (
                partial_values,
                partial_ids,
                tile_values,
                tile_ids,
                candidate_ids,
                output,
            )
        )
    ):
        raise ValueError("refined Markov workspaces have invalid shapes or dtypes")

    _markov_project_argmax_stage1[(triton.cdiv(batch_size, batch_block), chunks)](
        base,
        token_ids,
        candidate_embedding_weight,
        candidate_weight,
        partial_values,
        partial_ids,
        vocab_size,
        base.stride(0),
        token_ids.stride(0),
        candidate_embedding_weight.stride(0),
        candidate_weight.stride(0),
        partial_values.stride(0),
        chunks=chunks,
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=batch_block,
        MARKOV_RANK=candidate_rank,
        RANK_BLOCK=triton.next_power_of_2(candidate_rank),
        BLOCK_VOCAB=block_vocab,
        REVERSE=reverse,
        num_warps=2,
        num_stages=1,
    )
    _markov_partial_topk_tiles[(batch_size, num_tiles)](
        partial_values,
        partial_ids,
        tile_values,
        tile_ids,
        partial_values.stride(0),
        tile_values.stride(0),
        tile_values.stride(1),
        chunks=chunks,
        TILE_SIZE=tile_size,
        TOP_K=top_k,
        num_warps=8,
    )
    candidate_count = num_tiles * top_k
    _markov_tile_topk[(batch_size,)](
        tile_values,
        tile_ids,
        candidate_ids,
        tile_values.stride(0),
        candidate_ids.stride(0),
        candidate_count=candidate_count,
        REDUCE_SIZE=triton.next_power_of_2(candidate_count),
        TOP_K=top_k,
        num_warps=4,
    )
    _markov_refine_candidates[(batch_size,)](
        base,
        token_ids,
        exact_embedding_weight,
        exact_weight,
        candidate_ids,
        output,
        base.stride(0),
        token_ids.stride(0),
        exact_embedding_weight.stride(0),
        exact_weight.stride(0),
        candidate_ids.stride(0),
        output.stride(0),
        MARKOV_RANK=exact_rank,
        RANK_BLOCK=triton.next_power_of_2(exact_rank),
        TOP_K=top_k,
        num_warps=4,
    )
    return output[:batch_size]


@triton.jit
def _longspark_attention_kernel(
    query,
    global_keys,
    global_values,
    raw_keys,
    raw_values,
    raw_mask,
    raw_cache,
    raw_block_table,
    raw_seq_lens,
    local_keys,
    local_values,
    output,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_r: tl.constexpr,
    query_stride_d: tl.constexpr,
    global_key_stride_b: tl.constexpr,
    global_key_stride_h: tl.constexpr,
    global_key_stride_s: tl.constexpr,
    global_key_stride_d: tl.constexpr,
    global_value_stride_b: tl.constexpr,
    global_value_stride_h: tl.constexpr,
    global_value_stride_s: tl.constexpr,
    global_value_stride_d: tl.constexpr,
    raw_key_stride_b: tl.constexpr,
    raw_key_stride_h: tl.constexpr,
    raw_key_stride_s: tl.constexpr,
    raw_key_stride_d: tl.constexpr,
    raw_value_stride_b: tl.constexpr,
    raw_value_stride_h: tl.constexpr,
    raw_value_stride_s: tl.constexpr,
    raw_value_stride_d: tl.constexpr,
    raw_mask_stride_b: tl.constexpr,
    raw_mask_stride_s: tl.constexpr,
    raw_cache_stride_b: tl.constexpr,
    raw_cache_stride_h: tl.constexpr,
    raw_cache_stride_s: tl.constexpr,
    raw_cache_stride_d: tl.constexpr,
    raw_table_stride_b: tl.constexpr,
    raw_table_stride_s: tl.constexpr,
    raw_seq_len_stride: tl.constexpr,
    local_key_stride_b: tl.constexpr,
    local_key_stride_h: tl.constexpr,
    local_key_stride_s: tl.constexpr,
    local_key_stride_d: tl.constexpr,
    local_value_stride_b: tl.constexpr,
    local_value_stride_h: tl.constexpr,
    local_value_stride_s: tl.constexpr,
    local_value_stride_d: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_r: tl.constexpr,
    output_stride_d: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QUERY_GROUPS: tl.constexpr,
    QUERY_HEADS_PER_PROGRAM: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GLOBAL_ROWS: tl.constexpr,
    RAW_ROWS: tl.constexpr,
    LOCAL_ROWS: tl.constexpr,
    QUERY_ROW_BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    GLOBAL_ROW_BLOCK: tl.constexpr,
    RAW_ROW_BLOCK: tl.constexpr,
    LOCAL_ROW_BLOCK: tl.constexpr,
    RAW_BLOCK_SIZE: tl.constexpr,
    PAGED_RAW: tl.constexpr,
    SCALE: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    programs_per_batch = QUERY_HEADS // QUERY_HEADS_PER_PROGRAM
    batch = program // programs_per_batch
    head_program = program - batch * programs_per_batch
    query_head_base = head_program * QUERY_HEADS_PER_PROGRAM
    kv_head = query_head_base // QUERY_GROUPS
    rows = tl.arange(0, QUERY_ROW_BLOCK)
    dims = tl.arange(0, HEAD_BLOCK)
    raw_sources = tl.arange(0, RAW_ROW_BLOCK)
    global_sources = tl.arange(0, GLOBAL_ROW_BLOCK)
    local_sources = tl.arange(0, LOCAL_ROW_BLOCK)
    query_head_offsets = rows // QUERY_ROWS
    query_rows = rows - query_head_offsets * QUERY_ROWS
    query_heads = query_head_base + query_head_offsets
    valid_rows = rows < QUERY_HEADS_PER_PROGRAM * QUERY_ROWS
    valid_dims = dims < HEAD_DIM

    query_block = tl.load(
        query
        + batch * query_stride_b
        + query_heads[:, None] * query_stride_h
        + query_rows[:, None] * query_stride_r
        + dims[None, :] * query_stride_d,
        mask=valid_rows[:, None] & valid_dims[None, :],
        other=0.0,
    )

    if PAGED_RAW:
        raw_end = tl.load(raw_seq_lens + batch * raw_seq_len_stride)
        raw_start = tl.maximum(raw_end - RAW_ROWS, 0)
        raw_positions = raw_start + raw_sources
        valid_raw = (raw_sources < RAW_ROWS) & (raw_positions < raw_end)
        logical_blocks = raw_positions // RAW_BLOCK_SIZE
        offsets = raw_positions % RAW_BLOCK_SIZE
        physical_blocks = tl.load(
            raw_block_table
            + batch * raw_table_stride_b
            + logical_blocks * raw_table_stride_s,
            mask=valid_raw,
            other=0,
        )
        raw_key_block = tl.load(
            raw_cache
            + physical_blocks[None, :] * raw_cache_stride_b
            + kv_head * raw_cache_stride_h
            + offsets[None, :] * raw_cache_stride_s
            + dims[:, None] * raw_cache_stride_d,
            mask=valid_dims[:, None] & valid_raw[None, :],
            other=0.0,
        )
    else:
        valid_raw_source = raw_sources < RAW_ROWS
        raw_key_block = tl.load(
            raw_keys
            + batch * raw_key_stride_b
            + kv_head * raw_key_stride_h
            + raw_sources[None, :] * raw_key_stride_s
            + dims[:, None] * raw_key_stride_d,
            mask=valid_dims[:, None] & valid_raw_source[None, :],
            other=0.0,
        )
        valid_raw = tl.load(
            raw_mask + batch * raw_mask_stride_b + raw_sources * raw_mask_stride_s,
            mask=valid_raw_source,
            other=0,
        ).to(tl.int1)
    raw_scores = tl.dot(query_block, raw_key_block) * SCALE
    raw_scores = tl.where(valid_raw[None, :], raw_scores, -float("inf"))

    valid_global = global_sources < GLOBAL_ROWS
    global_scores = tl.full(
        (QUERY_ROW_BLOCK, GLOBAL_ROW_BLOCK), -float("inf"), tl.float32
    )
    # Global K/V are query-head specific, while raw and local K/V are shared
    # by every query head in a GQA group. Keep the global dot products
    # separate, but process several query heads together so the much larger
    # raw window is loaded only once.
    for head_offset in tl.static_range(QUERY_HEADS_PER_PROGRAM):
        global_key_block = tl.load(
            global_keys
            + batch * global_key_stride_b
            + (query_head_base + head_offset) * global_key_stride_h
            + global_sources[None, :] * global_key_stride_s
            + dims[:, None] * global_key_stride_d,
            mask=valid_dims[:, None] & valid_global[None, :],
            other=0.0,
        )
        head_scores = tl.dot(query_block, global_key_block) * SCALE
        global_scores = tl.where(
            (query_head_offsets == head_offset)[:, None],
            head_scores,
            global_scores,
        )
    global_scores = tl.where(valid_global[None, :], global_scores, -float("inf"))

    valid_local = local_sources < LOCAL_ROWS
    local_key_block = tl.load(
        local_keys
        + batch * local_key_stride_b
        + kv_head * local_key_stride_h
        + local_sources[None, :] * local_key_stride_s
        + dims[:, None] * local_key_stride_d,
        mask=valid_dims[:, None] & valid_local[None, :],
        other=0.0,
    )
    local_scores = tl.dot(query_block, local_key_block) * SCALE
    local_scores = tl.where(valid_local[None, :], local_scores, -float("inf"))

    row_max = tl.maximum(
        tl.max(raw_scores, axis=1),
        tl.maximum(tl.max(global_scores, axis=1), tl.max(local_scores, axis=1)),
    )
    raw_weights = tl.where(
        valid_raw[None, :], tl.exp(raw_scores - row_max[:, None]), 0.0
    )
    global_weights = tl.where(
        valid_global[None, :], tl.exp(global_scores - row_max[:, None]), 0.0
    )
    local_weights = tl.where(
        valid_local[None, :], tl.exp(local_scores - row_max[:, None]), 0.0
    )
    normalizer = (
        tl.sum(raw_weights, axis=1)
        + tl.sum(global_weights, axis=1)
        + tl.sum(local_weights, axis=1)
    )

    if PAGED_RAW:
        raw_value_block = tl.load(
            raw_cache
            + physical_blocks[:, None] * raw_cache_stride_b
            + kv_head * raw_cache_stride_h
            + offsets[:, None] * raw_cache_stride_s
            + (HEAD_DIM + dims[None, :]) * raw_cache_stride_d,
            mask=valid_raw[:, None] & valid_dims[None, :],
            other=0.0,
        )
    else:
        raw_value_block = tl.load(
            raw_values
            + batch * raw_value_stride_b
            + kv_head * raw_value_stride_h
            + raw_sources[:, None] * raw_value_stride_s
            + dims[None, :] * raw_value_stride_d,
            mask=valid_raw[:, None] & valid_dims[None, :],
            other=0.0,
        )
    local_value_block = tl.load(
        local_values
        + batch * local_value_stride_b
        + kv_head * local_value_stride_h
        + local_sources[:, None] * local_value_stride_s
        + dims[None, :] * local_value_stride_d,
        mask=valid_local[:, None] & valid_dims[None, :],
        other=0.0,
    )
    global_attended = tl.zeros((QUERY_ROW_BLOCK, HEAD_BLOCK), tl.float32)
    for head_offset in tl.static_range(QUERY_HEADS_PER_PROGRAM):
        global_value_block = tl.load(
            global_values
            + batch * global_value_stride_b
            + (query_head_base + head_offset) * global_value_stride_h
            + global_sources[:, None] * global_value_stride_s
            + dims[None, :] * global_value_stride_d,
            mask=valid_global[:, None] & valid_dims[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        head_weights = tl.where(
            (query_head_offsets == head_offset)[:, None], global_weights, 0.0
        )
        global_attended += tl.dot(head_weights.to(tl.bfloat16), global_value_block)
    attended = (
        tl.dot(raw_weights.to(tl.bfloat16), raw_value_block)
        + global_attended
        + tl.dot(local_weights.to(tl.bfloat16), local_value_block)
    ) / normalizer[:, None]
    tl.store(
        output
        + batch * output_stride_b
        + query_heads[:, None] * output_stride_h
        + query_rows[:, None] * output_stride_r
        + dims[None, :] * output_stride_d,
        attended,
        mask=valid_rows[:, None] & valid_dims[None, :],
    )


def _attention_query_heads_per_program(
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    query_rows: int,
    device: torch.device,
) -> int:
    """Pack GQA heads while retaining enough programs to occupy the GPU."""
    query_groups = query_heads // kv_heads
    max_heads = max(1, 32 // query_rows)
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    occupancy_floor = max(1, multiprocessors - multiprocessors // 8)
    for heads in range(min(query_groups, max_heads), 0, -1):
        if (
            query_groups % heads == 0
            and batch_size * query_heads // heads >= occupancy_floor
        ):
            return heads
    return 1


def longspark_attention(
    *,
    query: torch.Tensor,
    global_keys: torch.Tensor,
    global_values: torch.Tensor,
    raw_keys: torch.Tensor,
    raw_values: torch.Tensor,
    raw_mask: torch.Tensor,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run shape-parameterized Global + Raw + Local GQA without KV expansion."""
    if query.ndim != 4 or local_keys.ndim != 4:
        raise ValueError("query and local K/V must be rank-4 tensors")
    batch_size, query_heads, query_rows, head_dim = query.shape
    kv_heads = local_keys.shape[1]
    global_rows = global_keys.shape[2] if global_keys.ndim == 4 else 0
    raw_rows = raw_keys.shape[2] if raw_keys.ndim == 4 else 0
    local_rows = local_keys.shape[2]
    tensors = (
        global_keys,
        global_values,
        raw_keys,
        raw_values,
        raw_mask,
        local_keys,
        local_values,
    )
    if (
        not query.is_cuda
        or query.dtype != torch.bfloat16
        or min(query_heads, query_rows, head_dim, kv_heads, global_rows, raw_rows) <= 0
        or local_rows != query_rows
        or query_heads % kv_heads
        or global_keys.shape != (batch_size, query_heads, global_rows, head_dim)
        or global_keys.dtype != query.dtype
        or global_values.shape != global_keys.shape
        or global_values.dtype != torch.float32
        or raw_keys.shape != (batch_size, kv_heads, raw_rows, head_dim)
        or raw_values.shape != raw_keys.shape
        or raw_keys.dtype != query.dtype
        or raw_values.dtype != query.dtype
        or raw_mask.shape != (batch_size, raw_rows)
        or raw_mask.dtype != torch.bool
        or local_keys.shape != (batch_size, kv_heads, local_rows, head_dim)
        or local_values.shape != local_keys.shape
        or local_keys.dtype != query.dtype
        or local_values.dtype != query.dtype
        or any(tensor.device != query.device for tensor in tensors)
    ):
        raise ValueError("LongSpark attention tensors violate the kernel contract")
    output = torch.empty_like(query)
    query_groups = query_heads // kv_heads
    query_heads_per_program = _attention_query_heads_per_program(
        batch_size, query_heads, kv_heads, query_rows, query.device
    )
    grouped_query_rows = query_heads_per_program * query_rows
    _longspark_attention_kernel[(batch_size * query_heads // query_heads_per_program,)](
        query,
        global_keys,
        global_values,
        raw_keys,
        raw_values,
        raw_mask,
        raw_keys,
        raw_mask,
        raw_mask,
        local_keys,
        local_values,
        output,
        *query.stride(),
        *global_keys.stride(),
        *global_values.stride(),
        *raw_keys.stride(),
        *raw_values.stride(),
        *raw_mask.stride(),
        *raw_keys.stride(),
        *raw_mask.stride(),
        raw_mask.stride(0),
        *local_keys.stride(),
        *local_values.stride(),
        *output.stride(),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        QUERY_GROUPS=query_groups,
        QUERY_HEADS_PER_PROGRAM=query_heads_per_program,
        QUERY_ROWS=query_rows,
        HEAD_DIM=head_dim,
        GLOBAL_ROWS=global_rows,
        RAW_ROWS=raw_rows,
        LOCAL_ROWS=local_rows,
        QUERY_ROW_BLOCK=max(16, triton.next_power_of_2(grouped_query_rows)),
        HEAD_BLOCK=max(16, triton.next_power_of_2(head_dim)),
        GLOBAL_ROW_BLOCK=max(16, triton.next_power_of_2(global_rows)),
        RAW_ROW_BLOCK=max(16, triton.next_power_of_2(raw_rows)),
        LOCAL_ROW_BLOCK=max(16, triton.next_power_of_2(local_rows)),
        RAW_BLOCK_SIZE=1,
        PAGED_RAW=False,
        SCALE=scale,
        num_warps=4,
        num_stages=1,
    )
    return output


def longspark_paged_attention(
    *,
    query: torch.Tensor,
    global_keys: torch.Tensor,
    global_values: torch.Tensor,
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    raw_seq_lens: torch.Tensor,
    raw_rows: int,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Read a parameterized raw window directly from a paged target KV cache."""
    if query.ndim != 4 or local_keys.ndim != 4:
        raise ValueError("query and local K/V must be rank-4 tensors")
    batch_size, query_heads, query_rows, head_dim = query.shape
    kv_heads = local_keys.shape[1]
    global_rows = global_keys.shape[2] if global_keys.ndim == 4 else 0
    local_rows = local_keys.shape[2]
    tensors = (
        global_keys,
        global_values,
        raw_cache,
        raw_block_table,
        raw_seq_lens,
        local_keys,
        local_values,
    )
    if (
        not query.is_cuda
        or query.dtype != torch.bfloat16
        or min(query_heads, query_rows, head_dim, kv_heads, global_rows, raw_rows) <= 0
        or local_rows != query_rows
        or query_heads % kv_heads
        or global_keys.shape != (batch_size, query_heads, global_rows, head_dim)
        or global_keys.dtype != query.dtype
        or global_values.shape != global_keys.shape
        or global_values.dtype != torch.float32
        or raw_cache.ndim != 4
        or raw_cache.shape[1] != kv_heads
        or raw_cache.shape[-1] != 2 * head_dim
        or raw_cache.dtype != query.dtype
        or raw_block_table.ndim != 2
        or raw_block_table.shape[0] < batch_size
        or raw_seq_lens.shape != (batch_size,)
        or raw_seq_lens.dtype not in (torch.int32, torch.int64)
        or local_keys.shape != (batch_size, kv_heads, local_rows, head_dim)
        or local_values.shape != local_keys.shape
        or local_keys.dtype != query.dtype
        or local_values.dtype != query.dtype
        or any(tensor.device != query.device for tensor in tensors)
    ):
        raise ValueError(
            "LongSpark paged attention tensors violate the kernel contract"
        )
    output = torch.empty_like(query)
    query_groups = query_heads // kv_heads
    query_heads_per_program = _attention_query_heads_per_program(
        batch_size, query_heads, kv_heads, query_rows, query.device
    )
    grouped_query_rows = query_heads_per_program * query_rows
    _longspark_attention_kernel[(batch_size * query_heads // query_heads_per_program,)](
        query,
        global_keys,
        global_values,
        raw_cache,
        raw_cache,
        raw_seq_lens,
        raw_cache,
        raw_block_table,
        raw_seq_lens,
        local_keys,
        local_values,
        output,
        *query.stride(),
        *global_keys.stride(),
        *global_values.stride(),
        *raw_cache.stride(),
        *raw_cache.stride(),
        raw_seq_lens.stride(0),
        raw_seq_lens.stride(0),
        *raw_cache.stride(),
        *raw_block_table.stride(),
        raw_seq_lens.stride(0),
        *local_keys.stride(),
        *local_values.stride(),
        *output.stride(),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        QUERY_GROUPS=query_groups,
        QUERY_HEADS_PER_PROGRAM=query_heads_per_program,
        QUERY_ROWS=query_rows,
        HEAD_DIM=head_dim,
        GLOBAL_ROWS=global_rows,
        RAW_ROWS=raw_rows,
        LOCAL_ROWS=local_rows,
        QUERY_ROW_BLOCK=max(16, triton.next_power_of_2(grouped_query_rows)),
        HEAD_BLOCK=max(16, triton.next_power_of_2(head_dim)),
        GLOBAL_ROW_BLOCK=max(16, triton.next_power_of_2(global_rows)),
        RAW_ROW_BLOCK=max(16, triton.next_power_of_2(raw_rows)),
        LOCAL_ROW_BLOCK=max(16, triton.next_power_of_2(local_rows)),
        RAW_BLOCK_SIZE=raw_cache.shape[2],
        PAGED_RAW=True,
        SCALE=scale,
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _update_longspark_global_delta_kernel(
    query,
    cache,
    block_table,
    batch_indices,
    request_slots,
    starts,
    ends,
    state_output,
    state_lse,
    query_stride_h: tl.constexpr,
    query_stride_g: tl.constexpr,
    query_stride_d: tl.constexpr,
    cache_stride_b: tl.constexpr,
    cache_stride_h: tl.constexpr,
    cache_stride_s: tl.constexpr,
    cache_stride_d: tl.constexpr,
    table_stride_b: tl.constexpr,
    table_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_g: tl.constexpr,
    output_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_h: tl.constexpr,
    lse_stride_g: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GLOBAL_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_DELTA_ROWS: tl.constexpr,
    QUERY_ROW_BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    DELTA_ROW_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    request = program // KV_HEADS
    kv_head = program - request * KV_HEADS
    batch_index = tl.load(batch_indices + request)
    state_slot = tl.load(request_slots + request)
    start = tl.load(starts + request)
    end = tl.load(ends + request)

    rows = tl.arange(0, QUERY_ROW_BLOCK)
    dims = tl.arange(0, HEAD_BLOCK)
    positions = tl.arange(0, DELTA_ROW_BLOCK)
    query_groups = QUERY_HEADS // KV_HEADS
    query_head = kv_head * query_groups + rows // GLOBAL_ROWS
    global_row = rows % GLOBAL_ROWS
    valid_query_row = rows < query_groups * GLOBAL_ROWS
    valid_dim = dims < HEAD_DIM
    query_block = tl.load(
        query
        + query_head[:, None] * query_stride_h
        + global_row[:, None] * query_stride_g
        + dims[None, :] * query_stride_d,
        mask=valid_query_row[:, None] & valid_dim[None, :],
        other=0.0,
    )

    logical_position = start + positions
    valid = (positions < MAX_DELTA_ROWS) & (logical_position < end)
    logical_block = logical_position // BLOCK_SIZE
    offset = logical_position % BLOCK_SIZE
    physical_block = tl.load(
        block_table + batch_index * table_stride_b + logical_block * table_stride_s,
        mask=valid,
        other=0,
    )
    key_block = tl.load(
        cache
        + physical_block[None, :] * cache_stride_b
        + kv_head * cache_stride_h
        + offset[None, :] * cache_stride_s
        + dims[:, None] * cache_stride_d,
        mask=valid_dim[:, None] & valid[None, :],
        other=0.0,
    )
    scores = tl.dot(query_block, key_block)
    scores = scores.to(tl.bfloat16).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, -float("inf"))
    new_max = tl.max(scores, axis=1)
    weights = tl.exp(scores - new_max[:, None])
    weights = tl.where(valid[None, :], weights, 0.0)
    normalizer = tl.sum(weights, axis=1)
    new_lse = new_max + tl.log(normalizer)

    value_block = tl.load(
        cache
        + physical_block[:, None] * cache_stride_b
        + kv_head * cache_stride_h
        + offset[:, None] * cache_stride_s
        + (HEAD_DIM + dims[None, :]) * cache_stride_d,
        mask=valid[:, None] & valid_dim[None, :],
        other=0.0,
    )
    weighted_values = tl.dot(weights.to(tl.bfloat16), value_block)
    weighted_values = weighted_values.to(tl.bfloat16).to(tl.float32)
    new_output = weighted_values / normalizer[:, None]

    lse_offsets = (
        state_slot * lse_stride_b
        + query_head * lse_stride_h
        + global_row * lse_stride_g
    )
    old_lse = tl.load(
        state_lse + lse_offsets, mask=valid_query_row, other=-float("inf")
    )
    merged_max = tl.maximum(old_lse, new_lse)
    old_weight = tl.exp(old_lse - merged_max)
    new_weight = tl.exp(new_lse - merged_max)
    merged_normalizer = old_weight + new_weight
    merged_lse = merged_max + tl.log(merged_normalizer)

    output_offsets = (
        state_slot * output_stride_b
        + query_head[:, None] * output_stride_h
        + global_row[:, None] * output_stride_g
        + dims[None, :] * output_stride_d
    )
    old_output = tl.load(
        state_output + output_offsets,
        mask=valid_query_row[:, None] & valid_dim[None, :],
        other=0.0,
    )
    output = (
        old_output * old_weight[:, None] + new_output * new_weight[:, None]
    ) / merged_normalizer[:, None]
    tl.store(
        state_output + output_offsets,
        output,
        mask=valid_query_row[:, None] & valid_dim[None, :],
    )
    tl.store(state_lse + lse_offsets, merged_lse, mask=valid_query_row)


def update_longspark_global_delta(
    *,
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    batch_indices: torch.Tensor,
    request_slots: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
    max_delta_rows: int,
) -> None:
    """Update parameterized global-memory state from newly committed KV rows."""
    batch_size = batch_indices.numel()
    if query.ndim != 3 or cache.ndim != 4:
        raise ValueError("LongSpark global query/cache tensors have invalid ranks")
    query_heads, global_rows, head_dim = query.shape
    kv_heads = cache.shape[1]
    index_tensors = (batch_indices, request_slots, starts, ends)
    tensors = (*index_tensors, cache, block_table, state_output, state_lse)
    if (
        not query.is_cuda
        or query.dtype != torch.bfloat16
        or min(query_heads, global_rows, head_dim, kv_heads, max_delta_rows) <= 0
        or query_heads % kv_heads
        or cache.dtype != torch.bfloat16
        or cache.shape[-1] != 2 * head_dim
        or block_table.ndim != 2
        or batch_indices.shape != (batch_size,)
        or request_slots.shape != (batch_size,)
        or starts.shape != (batch_size,)
        or ends.shape != (batch_size,)
        or state_output.ndim != 4
        or state_output.shape[1:] != query.shape
        or state_output.dtype != torch.float32
        or state_lse.shape != state_output.shape[:-1]
        or state_lse.dtype != torch.float32
        or any(tensor.device != query.device for tensor in tensors)
        or any(not tensor.is_contiguous() for tensor in index_tensors)
    ):
        raise ValueError("LongSpark global delta tensors violate the kernel contract")
    if batch_size == 0:
        return
    query_rows = (query_heads // kv_heads) * global_rows
    _update_longspark_global_delta_kernel[(batch_size * kv_heads,)](
        query,
        cache,
        block_table,
        batch_indices,
        request_slots,
        starts,
        ends,
        state_output,
        state_lse,
        *query.stride(),
        *cache.stride(),
        *block_table.stride(),
        *state_output.stride(),
        *state_lse.stride(),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GLOBAL_ROWS=global_rows,
        HEAD_DIM=head_dim,
        MAX_DELTA_ROWS=max_delta_rows,
        QUERY_ROW_BLOCK=max(16, triton.next_power_of_2(query_rows)),
        HEAD_BLOCK=max(16, triton.next_power_of_2(head_dim)),
        DELTA_ROW_BLOCK=max(16, triton.next_power_of_2(max_delta_rows)),
        BLOCK_SIZE=cache.shape[2],
        SCALE=head_dim**-0.5,
        num_warps=8,
        num_stages=2,
    )


@triton.jit
def _update_longspark_global_delta_batched_kernel(
    queries,
    cache_ptrs,
    block_table_ptrs,
    batch_indices,
    request_slots,
    starts,
    ends,
    state_output,
    state_lse,
    batch_output,
    query_stride_l: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_g: tl.constexpr,
    query_stride_d: tl.constexpr,
    cache_stride_b: tl.constexpr,
    cache_stride_h: tl.constexpr,
    cache_stride_s: tl.constexpr,
    cache_stride_d: tl.constexpr,
    table_stride_b: tl.constexpr,
    table_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_l: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_g: tl.constexpr,
    output_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_l: tl.constexpr,
    lse_stride_h: tl.constexpr,
    lse_stride_g: tl.constexpr,
    batch_output_stride_b: tl.constexpr,
    batch_output_stride_l: tl.constexpr,
    batch_output_stride_h: tl.constexpr,
    batch_output_stride_g: tl.constexpr,
    batch_output_stride_d: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GLOBAL_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_DELTA_ROWS: tl.constexpr,
    QUERY_ROW_BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    DELTA_ROW_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    layer = tl.program_id(1)
    request = program // KV_HEADS
    kv_head = program - request * KV_HEADS
    batch_index = tl.load(batch_indices + request)
    state_slot = tl.load(request_slots + request)
    start = tl.load(starts + request)
    end = tl.load(ends + request)
    cache_address = tl.load(cache_ptrs + layer)
    cache = cache_address.to(tl.pointer_type(queries.dtype.element_ty))
    block_table_address = tl.load(block_table_ptrs + layer)
    block_table = block_table_address.to(tl.pointer_type(tl.int32))

    rows = tl.arange(0, QUERY_ROW_BLOCK)
    dims = tl.arange(0, HEAD_BLOCK)
    positions = tl.arange(0, DELTA_ROW_BLOCK)
    query_groups = QUERY_HEADS // KV_HEADS
    query_head = kv_head * query_groups + rows // GLOBAL_ROWS
    global_row = rows % GLOBAL_ROWS
    valid_query_row = rows < query_groups * GLOBAL_ROWS
    valid_dim = dims < HEAD_DIM
    query_block = tl.load(
        queries
        + layer * query_stride_l
        + query_head[:, None] * query_stride_h
        + global_row[:, None] * query_stride_g
        + dims[None, :] * query_stride_d,
        mask=valid_query_row[:, None] & valid_dim[None, :],
        other=0.0,
    )

    logical_position = start + positions
    valid = (positions < MAX_DELTA_ROWS) & (logical_position < end)
    logical_block = logical_position // BLOCK_SIZE
    offset = logical_position % BLOCK_SIZE
    physical_block = tl.load(
        block_table + batch_index * table_stride_b + logical_block * table_stride_s,
        mask=valid,
        other=0,
    )
    key_block = tl.load(
        cache
        + physical_block[None, :] * cache_stride_b
        + kv_head * cache_stride_h
        + offset[None, :] * cache_stride_s
        + dims[:, None] * cache_stride_d,
        mask=valid_dim[:, None] & valid[None, :],
        other=0.0,
    )
    scores = tl.dot(query_block, key_block)
    scores = scores.to(tl.bfloat16).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, -float("inf"))
    new_max = tl.max(scores, axis=1)
    weights = tl.exp(scores - new_max[:, None])
    weights = tl.where(valid[None, :], weights, 0.0)
    normalizer = tl.sum(weights, axis=1)
    new_lse = new_max + tl.log(normalizer)

    value_block = tl.load(
        cache
        + physical_block[:, None] * cache_stride_b
        + kv_head * cache_stride_h
        + offset[:, None] * cache_stride_s
        + (HEAD_DIM + dims[None, :]) * cache_stride_d,
        mask=valid[:, None] & valid_dim[None, :],
        other=0.0,
    )
    weighted_values = tl.dot(weights.to(tl.bfloat16), value_block)
    weighted_values = weighted_values.to(tl.bfloat16).to(tl.float32)
    new_output = weighted_values / normalizer[:, None]

    lse_offsets = (
        state_slot * lse_stride_b
        + layer * lse_stride_l
        + query_head * lse_stride_h
        + global_row * lse_stride_g
    )
    old_lse = tl.load(
        state_lse + lse_offsets, mask=valid_query_row, other=-float("inf")
    )
    merged_max = tl.maximum(old_lse, new_lse)
    old_weight = tl.exp(old_lse - merged_max)
    new_weight = tl.exp(new_lse - merged_max)
    merged_normalizer = old_weight + new_weight
    merged_lse = merged_max + tl.log(merged_normalizer)

    output_offsets = (
        state_slot * output_stride_b
        + layer * output_stride_l
        + query_head[:, None] * output_stride_h
        + global_row[:, None] * output_stride_g
        + dims[None, :] * output_stride_d
    )
    old_output = tl.load(
        state_output + output_offsets,
        mask=valid_query_row[:, None] & valid_dim[None, :],
        other=0.0,
    )
    output = (
        old_output * old_weight[:, None] + new_output * new_weight[:, None]
    ) / merged_normalizer[:, None]
    valid_output = valid_query_row[:, None] & valid_dim[None, :]
    tl.store(state_output + output_offsets, output, mask=valid_output)
    tl.store(state_lse + lse_offsets, merged_lse, mask=valid_query_row)
    batch_offsets = (
        request * batch_output_stride_b
        + layer * batch_output_stride_l
        + query_head[:, None] * batch_output_stride_h
        + global_row[:, None] * batch_output_stride_g
        + dims[None, :] * batch_output_stride_d
    )
    tl.store(batch_output + batch_offsets, output, mask=valid_output)


def update_longspark_global_delta_batched(
    *,
    queries: torch.Tensor,
    caches: tuple[torch.Tensor, ...],
    cache_ptrs: torch.Tensor,
    block_tables: tuple[torch.Tensor, ...],
    block_table_ptrs: torch.Tensor,
    batch_indices: torch.Tensor,
    request_slots: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
    batch_output: torch.Tensor,
    max_delta_rows: int,
) -> None:
    """Update all selected global-memory layers in one launch."""
    batch_size = batch_indices.numel()
    if queries.ndim != 4 or not caches or len(caches) != len(block_tables):
        raise ValueError("LongSpark batched global delta tensors have invalid ranks")
    num_layers, query_heads, global_rows, head_dim = queries.shape
    if len(caches) != num_layers:
        raise ValueError("LongSpark batched global delta layer count does not match")
    first_cache = caches[0]
    first_table = block_tables[0]
    kv_heads = first_cache.shape[1]
    index_tensors = (batch_indices, request_slots, starts, ends)
    pointer_tensors = (cache_ptrs, block_table_ptrs)
    if (
        not queries.is_cuda
        or queries.dtype != torch.bfloat16
        or min(
            num_layers,
            query_heads,
            global_rows,
            head_dim,
            kv_heads,
            max_delta_rows,
        )
        <= 0
        or query_heads % kv_heads
        or any(
            cache.dtype != torch.bfloat16
            or cache.device != queries.device
            or cache.ndim != 4
            or cache.shape != first_cache.shape
            or cache.stride() != first_cache.stride()
            or cache.shape[-1] != 2 * head_dim
            for cache in caches
        )
        or any(
            table.dtype != torch.int32
            or table.device != queries.device
            or table.ndim != 2
            or table.shape != first_table.shape
            or table.stride() != first_table.stride()
            for table in block_tables
        )
        or cache_ptrs.shape != (num_layers,)
        or block_table_ptrs.shape != (num_layers,)
        or any(
            tensor.dtype != torch.uint64
            or tensor.device != queries.device
            or not tensor.is_contiguous()
            for tensor in pointer_tensors
        )
        or any(tensor.shape != (batch_size,) for tensor in index_tensors)
        or any(
            tensor.device != queries.device or not tensor.is_contiguous()
            for tensor in index_tensors
        )
        or state_output.ndim != 5
        or state_output.shape[1:] != queries.shape
        or state_output.dtype != torch.float32
        or state_lse.shape != state_output.shape[:-1]
        or state_lse.dtype != torch.float32
        or batch_output.shape != (batch_size, *queries.shape)
        or batch_output.dtype != torch.float32
        or any(
            tensor.device != queries.device
            for tensor in (state_output, state_lse, batch_output)
        )
    ):
        raise ValueError("LongSpark batched global delta tensors violate the contract")
    if batch_size == 0:
        return
    query_rows = (query_heads // kv_heads) * global_rows
    _update_longspark_global_delta_batched_kernel[(batch_size * kv_heads, num_layers)](
        queries,
        cache_ptrs,
        block_table_ptrs,
        batch_indices,
        request_slots,
        starts,
        ends,
        state_output,
        state_lse,
        batch_output,
        *queries.stride(),
        *first_cache.stride(),
        *first_table.stride(),
        *state_output.stride(),
        *state_lse.stride(),
        *batch_output.stride(),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GLOBAL_ROWS=global_rows,
        HEAD_DIM=head_dim,
        MAX_DELTA_ROWS=max_delta_rows,
        QUERY_ROW_BLOCK=max(16, triton.next_power_of_2(query_rows)),
        HEAD_BLOCK=max(16, triton.next_power_of_2(head_dim)),
        DELTA_ROW_BLOCK=max(16, triton.next_power_of_2(max_delta_rows)),
        BLOCK_SIZE=first_cache.shape[2],
        SCALE=head_dim**-0.5,
        num_warps=8,
        num_stages=2,
    )


__all__ = [
    "fused_add_argmax",
    "fused_markov_argmax",
    "fused_markov_refined_argmax",
    "longspark_attention",
    "longspark_paged_attention",
    "prepare_longspark_shifted_input",
    "update_longspark_global_delta",
    "update_longspark_global_delta_batched",
]
