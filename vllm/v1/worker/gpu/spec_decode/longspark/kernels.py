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


@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _markov_project_argmax_stage1(
    base,
    embedding,
    weight,
    partial_values,
    partial_ids,
    vocab_size,
    base_stride_b: tl.constexpr,
    embedding_stride_b: tl.constexpr,
    weight_stride_v: tl.constexpr,
    chunks: tl.constexpr,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    MARKOV_RANK: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
) -> None:
    batch_group = tl.program_id(0)
    chunk_index = tl.program_id(1)
    batch_offsets = batch_group * BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
    rank_offsets = tl.arange(0, RANK_BLOCK)
    vocab_offsets = chunk_index * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    valid_batch = batch_offsets < BATCH_SIZE
    valid_rank = rank_offsets < MARKOV_RANK
    valid_vocab = vocab_offsets < vocab_size

    embeddings = tl.load(
        embedding + batch_offsets[:, None] * embedding_stride_b + rank_offsets[None, :],
        mask=valid_batch[:, None] & valid_rank[None, :],
        other=0.0,
    )
    weights = tl.load(
        weight + rank_offsets[:, None] + vocab_offsets[None, :] * weight_stride_v,
        mask=valid_rank[:, None] & valid_vocab[None, :],
        other=0.0,
    )
    bias = tl.dot(embeddings, weights).to(tl.bfloat16).to(tl.float32)
    base_values = tl.load(
        base + batch_offsets[:, None] * base_stride_b + vocab_offsets[None, :],
        mask=valid_batch[:, None] & valid_vocab[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    scores = base_values + bias
    best_values = tl.max(scores, axis=1)
    best_ids = tl.min(
        tl.where(scores == best_values[:, None], vocab_offsets[None, :], vocab_size),
        axis=1,
    )
    output_offsets = batch_offsets * chunks + chunk_index
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
    output_offset = batch_index * chunks + chunk_index
    tl.store(partial_values + output_offset, best_value)
    tl.store(partial_ids + output_offset, best_id)


@triton.jit
def _add_argmax_stage2(
    partial_values,
    partial_ids,
    output,
    vocab_size,
    output_stride: tl.constexpr,
    chunks: tl.constexpr,
    REDUCE_BLOCK_SIZE: tl.constexpr,
) -> None:
    batch_index = tl.program_id(0)
    chunk_offsets = tl.arange(0, REDUCE_BLOCK_SIZE)
    valid = chunk_offsets < chunks
    input_offsets = batch_index * chunks + chunk_offsets
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
        chunks=chunks,
        BLOCK_SIZE=block_size,
    )
    _add_argmax_stage2[(batch_size,)](
        partial_values,
        partial_ids,
        output,
        vocab_size,
        output.stride(0),
        chunks=chunks,
        REDUCE_BLOCK_SIZE=triton.next_power_of_2(chunks),
    )
    return output[:batch_size]


def fused_markov_argmax(
    base: torch.Tensor,
    embedding: torch.Tensor,
    weight: torch.Tensor,
    *,
    partial_values: torch.Tensor,
    partial_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fuse a rank-R Markov projection, base-logit add, and argmax."""
    if base.ndim != 2:
        raise ValueError("base must have shape [batch, vocab]")
    batch_size, vocab_size = base.shape
    if batch_size <= 0:
        raise ValueError("fused Markov projection requires a non-empty batch")
    if embedding.ndim != 2 or embedding.shape[0] != batch_size:
        raise ValueError("embedding must have shape [batch, rank]")
    markov_rank = embedding.shape[1]
    if markov_rank <= 0 or weight.shape != (vocab_size, markov_rank):
        raise ValueError("weight must have shape [vocab, rank]")
    if (
        not base.is_cuda
        or embedding.device != base.device
        or weight.device != base.device
        or base.dtype != torch.bfloat16
        or embedding.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or base.stride(1) != 1
        or embedding.stride(1) != 1
        or weight.stride(1) != 1
    ):
        raise ValueError("fused Markov projection requires contiguous BF16 CUDA rows")

    block_vocab = 64
    chunks = triton.cdiv(vocab_size, block_vocab)
    if (
        partial_values.shape[0] < batch_size
        or partial_values.shape[1] != chunks
        or partial_values.dtype != torch.float32
        or not partial_values.is_contiguous()
        or partial_values.device != base.device
        or partial_ids.shape[0] < batch_size
        or partial_ids.shape[1] != chunks
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

    batch_block = 16
    rank_block = max(16, triton.next_power_of_2(markov_rank))
    _markov_project_argmax_stage1[(triton.cdiv(batch_size, batch_block), chunks)](
        base,
        embedding,
        weight,
        partial_values,
        partial_ids,
        vocab_size,
        base.stride(0),
        embedding.stride(0),
        weight.stride(0),
        chunks=chunks,
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=batch_block,
        MARKOV_RANK=markov_rank,
        RANK_BLOCK=rank_block,
        BLOCK_VOCAB=block_vocab,
        num_warps=4,
        num_stages=1,
    )
    _add_argmax_stage2[(batch_size,)](
        partial_values,
        partial_ids,
        output,
        vocab_size,
        output.stride(0),
        chunks=chunks,
        REDUCE_BLOCK_SIZE=triton.next_power_of_2(chunks),
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
    batch = program // QUERY_HEADS
    query_head = program - batch * QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)
    rows = tl.arange(0, QUERY_ROW_BLOCK)
    dims = tl.arange(0, HEAD_BLOCK)
    raw_sources = tl.arange(0, RAW_ROW_BLOCK)
    global_sources = tl.arange(0, GLOBAL_ROW_BLOCK)
    local_sources = tl.arange(0, LOCAL_ROW_BLOCK)
    valid_rows = rows < QUERY_ROWS
    valid_dims = dims < HEAD_DIM

    query_block = tl.load(
        query
        + batch * query_stride_b
        + query_head * query_stride_h
        + rows[:, None] * query_stride_r
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
    global_key_block = tl.load(
        global_keys
        + batch * global_key_stride_b
        + query_head * global_key_stride_h
        + global_sources[None, :] * global_key_stride_s
        + dims[:, None] * global_key_stride_d,
        mask=valid_dims[:, None] & valid_global[None, :],
        other=0.0,
    )
    global_scores = tl.dot(query_block, global_key_block) * SCALE
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
    global_value_block = tl.load(
        global_values
        + batch * global_value_stride_b
        + query_head * global_value_stride_h
        + global_sources[:, None] * global_value_stride_s
        + dims[None, :] * global_value_stride_d,
        mask=valid_global[:, None] & valid_dims[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    local_value_block = tl.load(
        local_values
        + batch * local_value_stride_b
        + kv_head * local_value_stride_h
        + local_sources[:, None] * local_value_stride_s
        + dims[None, :] * local_value_stride_d,
        mask=valid_local[:, None] & valid_dims[None, :],
        other=0.0,
    )
    attended = (
        tl.dot(raw_weights.to(tl.bfloat16), raw_value_block)
        + tl.dot(global_weights.to(tl.bfloat16), global_value_block)
        + tl.dot(local_weights.to(tl.bfloat16), local_value_block)
    ) / normalizer[:, None]
    tl.store(
        output
        + batch * output_stride_b
        + query_head * output_stride_h
        + rows[:, None] * output_stride_r
        + dims[None, :] * output_stride_d,
        attended,
        mask=valid_rows[:, None] & valid_dims[None, :],
    )


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
    _longspark_attention_kernel[(batch_size * query_heads,)](
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
        QUERY_ROWS=query_rows,
        HEAD_DIM=head_dim,
        GLOBAL_ROWS=global_rows,
        RAW_ROWS=raw_rows,
        LOCAL_ROWS=local_rows,
        QUERY_ROW_BLOCK=max(16, triton.next_power_of_2(query_rows)),
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
    _longspark_attention_kernel[(batch_size * query_heads,)](
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
        QUERY_ROWS=query_rows,
        HEAD_DIM=head_dim,
        GLOBAL_ROWS=global_rows,
        RAW_ROWS=raw_rows,
        LOCAL_ROWS=local_rows,
        QUERY_ROW_BLOCK=max(16, triton.next_power_of_2(query_rows)),
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


__all__ = [
    "fused_add_argmax",
    "fused_markov_argmax",
    "longspark_attention",
    "longspark_paged_attention",
    "update_longspark_global_delta",
]
