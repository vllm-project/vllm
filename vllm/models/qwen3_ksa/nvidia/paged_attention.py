# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton paged-source attention for KSA query ranges."""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton


@triton.jit
def _ksa_paged_source_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    row_to_request_ptr,
    kv_start_ptr,
    kv_end_ptr,
    output_ptr,
    lse_ptr,
    softmax_scale,
    stride_q_row: tl.int64,
    stride_q_head: tl.int64,
    stride_q_dim: tl.int64,
    stride_k_block: tl.int64,
    stride_k_state: tl.int64,
    stride_k_head: tl.int64,
    stride_k_dim: tl.int64,
    stride_v_block: tl.int64,
    stride_v_state: tl.int64,
    stride_v_head: tl.int64,
    stride_v_dim: tl.int64,
    stride_block_table_request: tl.int64,
    stride_block_table_block: tl.int64,
    stride_output_row: tl.int64,
    stride_output_head: tl.int64,
    stride_output_dim: tl.int64,
    stride_lse_head: tl.int64,
    stride_lse_row: tl.int64,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    STATES_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // KV_GROUP_SIZE

    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    query_offsets = (
        row_idx * stride_q_row
        + query_head_idx * stride_q_head
        + dim_offsets * stride_q_dim
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    request_idx = tl.load(row_to_request_ptr + row_idx).to(tl.int64)
    kv_start = tl.load(kv_start_ptr + row_idx).to(tl.int64)
    kv_end = tl.load(kv_end_ptr + row_idx).to(tl.int64)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)

    if kv_end > kv_start:
        for block_start in tl.range(kv_start, kv_end, BLOCK_N):
            state_positions = block_start + tl.arange(0, BLOCK_N)
            state_mask = state_positions < kv_end
            logical_blocks = state_positions // STATES_PER_BLOCK
            physical_blocks = tl.load(
                block_table_ptr
                + request_idx * stride_block_table_request
                + logical_blocks * stride_block_table_block,
                mask=state_mask,
                other=0,
            ).to(tl.int64)
            state_offsets = state_positions % STATES_PER_BLOCK

            key_offsets = (
                physical_blocks[:, None] * stride_k_block
                + state_offsets[:, None] * stride_k_state
                + kv_head_idx * stride_k_head
                + dim_offsets[None, :] * stride_k_dim
            )
            key = tl.load(
                key_cache_ptr + key_offsets,
                mask=state_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
            scores = tl.where(state_mask, scores, -float("inf"))

            value_offsets = (
                physical_blocks[:, None] * stride_v_block
                + state_offsets[:, None] * stride_v_state
                + kv_head_idx * stride_v_head
                + dim_offsets[None, :] * stride_v_dim
            )
            value = tl.load(
                value_cache_ptr + value_offsets,
                mask=state_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            next_max = tl.maximum(tl.max(scores, axis=0), running_max)
            previous_scale = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max)
            accumulator = accumulator * previous_scale + tl.sum(
                probabilities[:, None] * value,
                axis=0,
            )
            running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=0)
            running_max = next_max

        output = accumulator / running_sum
        lse = running_max + tl.log(running_sum)
    else:
        output = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)
        lse = -float("inf")

    output_offsets = (
        row_idx * stride_output_row
        + query_head_idx * stride_output_head
        + dim_offsets * stride_output_dim
    )
    tl.store(output_ptr + output_offsets, output, mask=dim_mask)
    tl.store(
        lse_ptr + query_head_idx * stride_lse_head + row_idx * stride_lse_row,
        lse,
    )


@triton.jit
def _ksa_paged_source_split_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    row_to_request_ptr,
    kv_start_ptr,
    kv_end_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    softmax_scale,
    stride_q_row: tl.int64,
    stride_q_head: tl.int64,
    stride_q_dim: tl.int64,
    stride_k_block: tl.int64,
    stride_k_state: tl.int64,
    stride_k_head: tl.int64,
    stride_k_dim: tl.int64,
    stride_v_block: tl.int64,
    stride_v_state: tl.int64,
    stride_v_head: tl.int64,
    stride_v_dim: tl.int64,
    stride_block_table_request: tl.int64,
    stride_block_table_block: tl.int64,
    stride_partial_split: tl.int64,
    stride_partial_row: tl.int64,
    stride_partial_head: tl.int64,
    stride_partial_dim: tl.int64,
    stride_partial_lse_split: tl.int64,
    stride_partial_lse_head: tl.int64,
    stride_partial_lse_row: tl.int64,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    STATES_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_head_idx = query_head_idx // KV_GROUP_SIZE

    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    query_offsets = (
        row_idx * stride_q_row
        + query_head_idx * stride_q_head
        + dim_offsets * stride_q_dim
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    request_idx = tl.load(row_to_request_ptr + row_idx).to(tl.int64)
    source_start = tl.load(kv_start_ptr + row_idx).to(tl.int64)
    source_end = tl.load(kv_end_ptr + row_idx).to(tl.int64)
    source_length = source_end - source_start
    split_length = (source_length + NUM_SPLITS - 1) // NUM_SPLITS
    split_start = source_start + split_idx * split_length
    split_end = tl.minimum(split_start + split_length, source_end)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)

    if split_end > split_start:
        for block_start in tl.range(split_start, split_end, BLOCK_N):
            state_positions = block_start + tl.arange(0, BLOCK_N)
            state_mask = state_positions < split_end
            logical_blocks = state_positions // STATES_PER_BLOCK
            physical_blocks = tl.load(
                block_table_ptr
                + request_idx * stride_block_table_request
                + logical_blocks * stride_block_table_block,
                mask=state_mask,
                other=0,
            ).to(tl.int64)
            state_offsets = state_positions % STATES_PER_BLOCK

            key_offsets = (
                physical_blocks[:, None] * stride_k_block
                + state_offsets[:, None] * stride_k_state
                + kv_head_idx * stride_k_head
                + dim_offsets[None, :] * stride_k_dim
            )
            key = tl.load(
                key_cache_ptr + key_offsets,
                mask=state_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
            scores = tl.where(state_mask, scores, -float("inf"))

            value_offsets = (
                physical_blocks[:, None] * stride_v_block
                + state_offsets[:, None] * stride_v_state
                + kv_head_idx * stride_v_head
                + dim_offsets[None, :] * stride_v_dim
            )
            value = tl.load(
                value_cache_ptr + value_offsets,
                mask=state_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            next_max = tl.maximum(tl.max(scores, axis=0), running_max)
            previous_scale = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max)
            accumulator = accumulator * previous_scale + tl.sum(
                probabilities[:, None] * value,
                axis=0,
            )
            running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=0)
            running_max = next_max

        output = accumulator / running_sum
        lse = running_max + tl.log(running_sum)
    else:
        output = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)
        lse = -float("inf")

    partial_output_offsets = (
        split_idx * stride_partial_split
        + row_idx * stride_partial_row
        + query_head_idx * stride_partial_head
        + dim_offsets * stride_partial_dim
    )
    tl.store(
        partial_output_ptr + partial_output_offsets,
        output,
        mask=dim_mask,
    )
    tl.store(
        partial_lse_ptr
        + split_idx * stride_partial_lse_split
        + query_head_idx * stride_partial_lse_head
        + row_idx * stride_partial_lse_row,
        lse,
    )


@triton.jit
def _ksa_merge_split_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    lse_ptr,
    stride_partial_split: tl.int64,
    stride_partial_row: tl.int64,
    stride_partial_head: tl.int64,
    stride_partial_dim: tl.int64,
    stride_partial_lse_split: tl.int64,
    stride_partial_lse_head: tl.int64,
    stride_partial_lse_row: tl.int64,
    stride_output_row: tl.int64,
    stride_output_head: tl.int64,
    stride_output_dim: tl.int64,
    stride_lse_head: tl.int64,
    stride_lse_row: tl.int64,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    split_offsets = tl.arange(0, NUM_SPLITS)
    partial_lse_offsets = (
        split_offsets * stride_partial_lse_split
        + query_head_idx * stride_partial_lse_head
        + row_idx * stride_partial_lse_row
    )
    partial_lse = tl.load(partial_lse_ptr + partial_lse_offsets)
    maximum = tl.max(partial_lse, axis=0)
    has_source = maximum > -float("inf")
    safe_maximum = tl.where(has_source, maximum, 0.0)
    weights = tl.where(
        partial_lse > -float("inf"),
        tl.exp(partial_lse - safe_maximum),
        0.0,
    )
    weight_sum = tl.sum(weights, axis=0)

    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    partial_output_offsets = (
        split_offsets[:, None] * stride_partial_split
        + row_idx * stride_partial_row
        + query_head_idx * stride_partial_head
        + dim_offsets[None, :] * stride_partial_dim
    )
    partial_output = tl.load(
        partial_output_ptr + partial_output_offsets,
        mask=dim_mask[None, :],
        other=0.0,
    )
    output = tl.sum(weights[:, None] * partial_output, axis=0)
    output = tl.where(
        has_source,
        output / tl.where(has_source, weight_sum, 1.0),
        0.0,
    )

    output_offsets = (
        row_idx * stride_output_row
        + query_head_idx * stride_output_head
        + dim_offsets * stride_output_dim
    )
    tl.store(output_ptr + output_offsets, output, mask=dim_mask)
    tl.store(
        lse_ptr + query_head_idx * stride_lse_head + row_idx * stride_lse_row,
        tl.where(
            has_source,
            safe_maximum + tl.log(weight_sum),
            -float("inf"),
        ),
    )


@triton.jit
def _ksa_paged_source_tiled_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    query_start_loc_ptr,
    kv_start_ptr,
    kv_end_ptr,
    output_ptr,
    lse_ptr,
    softmax_scale,
    stride_q_row: tl.int64,
    stride_q_head: tl.int64,
    stride_q_dim: tl.int64,
    stride_k_block: tl.int64,
    stride_k_state: tl.int64,
    stride_k_head: tl.int64,
    stride_k_dim: tl.int64,
    stride_v_block: tl.int64,
    stride_v_state: tl.int64,
    stride_v_head: tl.int64,
    stride_v_dim: tl.int64,
    stride_block_table_request: tl.int64,
    stride_block_table_block: tl.int64,
    stride_output_row: tl.int64,
    stride_output_head: tl.int64,
    stride_output_dim: tl.int64,
    stride_lse_head: tl.int64,
    stride_lse_row: tl.int64,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    STATES_PER_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    request_idx = tl.program_id(0)
    query_block_idx = tl.program_id(1)
    query_head_idx = tl.program_id(2)
    kv_head_idx = query_head_idx // KV_GROUP_SIZE

    request_query_start = tl.load(query_start_loc_ptr + request_idx)
    request_query_end = tl.load(query_start_loc_ptr + request_idx + 1)
    query_block_start = request_query_start + query_block_idx * BLOCK_M
    if query_block_start >= request_query_end:
        return

    row_offsets = tl.arange(0, BLOCK_M)
    row_indices = query_block_start + row_offsets
    row_mask = row_indices < request_query_end
    source_starts = tl.load(
        kv_start_ptr + row_indices,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    source_ends = tl.load(
        kv_end_ptr + row_indices,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    tile_start = tl.min(
        tl.where(row_mask, source_starts, 2147483647),
        axis=0,
    )
    tile_end = tl.max(tl.where(row_mask, source_ends, 0), axis=0)

    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    query_offsets = (
        row_indices[:, None] * stride_q_row
        + query_head_idx * stride_q_head
        + dim_offsets[None, :] * stride_q_dim
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, PADDED_HEAD_DIM], dtype=tl.float32)

    if tile_end > tile_start:
        for block_start in tl.range(tile_start, tile_end, BLOCK_N):
            state_positions = block_start + tl.arange(0, BLOCK_N)
            state_mask = state_positions < tile_end
            logical_blocks = state_positions // STATES_PER_BLOCK
            physical_blocks = tl.load(
                block_table_ptr
                + request_idx * stride_block_table_request
                + logical_blocks * stride_block_table_block,
                mask=state_mask,
                other=0,
            ).to(tl.int64)
            state_offsets = state_positions % STATES_PER_BLOCK

            key_offsets = (
                physical_blocks[None, :] * stride_k_block
                + state_offsets[None, :] * stride_k_state
                + kv_head_idx * stride_k_head
                + dim_offsets[:, None] * stride_k_dim
            )
            key = tl.load(
                key_cache_ptr + key_offsets,
                mask=dim_mask[:, None] & state_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(query, key) * softmax_scale
            visibility = (
                row_mask[:, None]
                & state_mask[None, :]
                & (state_positions[None, :] >= source_starts[:, None])
                & (state_positions[None, :] < source_ends[:, None])
            )
            scores = tl.where(visibility, scores, -float("inf"))

            value_offsets = (
                physical_blocks[:, None] * stride_v_block
                + state_offsets[:, None] * stride_v_state
                + kv_head_idx * stride_v_head
                + dim_offsets[None, :] * stride_v_dim
            )
            value = tl.load(
                value_cache_ptr + value_offsets,
                mask=state_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            next_max = tl.maximum(tl.max(scores, axis=1), running_max)
            previous_scale = tl.where(
                running_sum > 0,
                tl.exp(running_max - next_max),
                0.0,
            )
            probabilities = tl.where(
                visibility,
                tl.exp(scores - next_max[:, None]),
                0.0,
            )
            accumulator = accumulator * previous_scale[:, None] + tl.dot(
                probabilities.to(value.dtype),
                value,
            )
            running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
            running_max = next_max

    has_source = running_sum > 0
    safe_sum = tl.where(has_source, running_sum, 1.0)
    output = accumulator / safe_sum[:, None]
    output = tl.where(has_source[:, None], output, 0.0)
    lse = tl.where(
        has_source,
        running_max + tl.log(safe_sum),
        -float("inf"),
    )

    output_offsets = (
        row_indices[:, None] * stride_output_row
        + query_head_idx * stride_output_head
        + dim_offsets[None, :] * stride_output_dim
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=row_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        lse_ptr + query_head_idx * stride_lse_head + row_indices * stride_lse_row,
        lse,
        mask=row_mask,
    )


def ksa_paged_source_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_to_request: torch.Tensor,
    kv_start: torch.Tensor,
    kv_end: torch.Tensor,
    softmax_scale: float,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend packed query rows to independent paged-cache ranges."""
    if query.ndim != 3:
        raise ValueError("KSA query must have shape [rows, heads, dim]")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("KSA caches must have shape [blocks, states, heads, dim]")
    row_count, num_query_heads, head_dim = query.shape
    if row_to_request.shape != (row_count,):
        raise ValueError("KSA request mapping must contain one item per row")
    if kv_start.shape != (row_count,) or kv_end.shape != (row_count,):
        raise ValueError("KSA source ranges must contain one item per row")
    num_kv_heads = key_cache.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("KSA query heads must be divisible by KV heads")
    if key_cache.shape[3] != head_dim:
        raise ValueError("KSA query and cache head dimensions must match")

    output = torch.empty_like(query)
    lse = torch.empty(
        (num_query_heads, row_count),
        dtype=torch.float32,
        device=query.device,
    )
    if row_count == 0:
        return output, lse

    padded_head_dim = triton.next_power_of_2(head_dim)
    if (query_start_loc is None) != (max_query_len is None):
        raise ValueError(
            "KSA tiled attention requires query_start_loc and max_query_len"
        )
    source_capacity = block_table.shape[1] * key_cache.shape[1]
    num_splits = 0
    if max_query_len == 1:
        if source_capacity >= 32768:
            num_splits = 16
        elif source_capacity >= 4096:
            num_splits = 8
    if num_splits:
        partial_output = torch.empty(
            (num_splits, row_count, num_query_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        partial_lse = torch.empty(
            (num_splits, num_query_heads, row_count),
            dtype=torch.float32,
            device=query.device,
        )
        _ksa_paged_source_split_kernel[(row_count, num_query_heads, num_splits)](
            query,
            key_cache,
            value_cache,
            block_table,
            row_to_request,
            kv_start,
            kv_end,
            partial_output,
            partial_lse,
            softmax_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_output.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            KV_GROUP_SIZE=num_query_heads // num_kv_heads,
            HEAD_DIM=head_dim,
            PADDED_HEAD_DIM=padded_head_dim,
            STATES_PER_BLOCK=key_cache.shape[1],
            BLOCK_N=32,
            NUM_SPLITS=num_splits,
            num_warps=4,
            num_stages=2,
        )
        _ksa_merge_split_kernel[(row_count, num_query_heads)](
            partial_output,
            partial_lse,
            output,
            lse,
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_output.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            HEAD_DIM=head_dim,
            PADDED_HEAD_DIM=padded_head_dim,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
        return output, lse
    # The tiled kernel uses tensor-core reductions whose rounding depends on
    # the prefill tile shape. Use the row-wise kernel in batch-invariant mode
    # so every token follows the same reduction order regardless of scheduling.
    if query_start_loc is not None and not envs.VLLM_BATCH_INVARIANT:
        num_reqs = query_start_loc.shape[0] - 1
        _ksa_paged_source_tiled_kernel[
            (num_reqs, triton.cdiv(max_query_len, 16), num_query_heads)
        ](
            query,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            kv_start,
            kv_end,
            output,
            lse,
            softmax_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            KV_GROUP_SIZE=num_query_heads // num_kv_heads,
            HEAD_DIM=head_dim,
            PADDED_HEAD_DIM=padded_head_dim,
            STATES_PER_BLOCK=key_cache.shape[1],
            BLOCK_M=16,
            BLOCK_N=32,
            num_warps=4,
            num_stages=2,
        )
        return output, lse

    _ksa_paged_source_kernel[(row_count, num_query_heads)](
        query,
        key_cache,
        value_cache,
        block_table,
        row_to_request,
        kv_start,
        kv_end,
        output,
        lse,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        KV_GROUP_SIZE=num_query_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        PADDED_HEAD_DIM=padded_head_dim,
        STATES_PER_BLOCK=key_cache.shape[1],
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )
    return output, lse


__all__ = ["ksa_paged_source_attention"]
