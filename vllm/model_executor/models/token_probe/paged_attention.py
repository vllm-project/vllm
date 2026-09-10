# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.triton_utils import tl, triton

MAX_SPLITS = 16
_target_blocks: int | None = None


@triton.jit
def _attend(
    q_ptr,
    kv_ptr,
    block_table_ptr,
    positions_ptr,
    query_offsets,
    query_mask,
    request_idx,
    key_lo,
    key_hi,
    q_stride,
    kv_stride,
    table_stride,
    softmax_scale,
    value_offset: tl.constexpr,
    window: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    query_head_offset,
):
    dim_offsets = tl.arange(0, head_dim)
    positions = tl.load(positions_ptr + query_offsets, mask=query_mask, other=0).to(
        tl.int64
    )
    query = tl.load(
        q_ptr
        + query_offsets[:, None] * q_stride
        + query_head_offset
        + dim_offsets[None, :],
        mask=query_mask[:, None],
        other=0.0,
    )
    running_max = tl.full([block_m], float("-inf"), tl.float32)
    running_sum = tl.zeros([block_m], tl.float32)
    accumulator = tl.zeros([block_m, head_dim], tl.float32)
    for start in range(key_lo, key_hi, block_n):
        key_offsets = start + tl.arange(0, block_n)
        key_mask = key_offsets < key_hi
        block_ids = tl.load(
            block_table_ptr + request_idx * table_stride + key_offsets // block_size,
            mask=key_mask,
            other=0,
        ).to(tl.int64)
        slots = block_ids * block_size + key_offsets % block_size
        cache_row = kv_ptr + slots[:, None] * kv_stride + dim_offsets[None, :]
        keys = tl.load(cache_row, mask=key_mask[:, None], other=0.0)
        scores = tl.dot(query, tl.trans(keys)).to(tl.float32) * softmax_scale
        visible = (
            (key_offsets[None, :] <= positions[:, None])
            & key_mask[None, :]
            & query_mask[:, None]
        )
        if window > 0:
            visible &= key_offsets[None, :] > positions[:, None] - window
        scores = tl.where(visible, scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.exp(running_max - safe_max)
        probabilities = tl.exp(scores - safe_max[:, None])
        running_sum = running_sum * alpha + tl.sum(probabilities, 1)
        values = tl.load(cache_row + value_offset, mask=key_mask[:, None], other=0.0)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities.to(values.dtype), values
        ).to(tl.float32)
        running_max = new_max
    return running_max, running_sum, accumulator


@triton.jit
def _bounds(positions_ptr, query_offsets, query_mask, window: tl.constexpr):
    positions = tl.load(positions_ptr + query_offsets, mask=query_mask, other=0).to(
        tl.int64
    )
    key_hi = tl.max(positions) + 1
    key_lo = tl.zeros([], tl.int64)
    if window > 0:
        min_position = tl.min(tl.where(query_mask, positions, 1 << 62))
        key_lo = tl.maximum(min_position + 1 - window, 0)
    return key_lo, key_hi


@triton.jit
def _probe_attention_kernel(
    q_ptr,
    kv_ptr,
    output_ptr,
    block_table_ptr,
    positions_ptr,
    query_start_ptr,
    q_stride,
    kv_stride,
    output_stride,
    table_stride,
    softmax_scale,
    value_offset: tl.constexpr,
    window: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    request_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    query_block_idx = tl.program_id(2)
    query_start = tl.load(query_start_ptr + request_idx).to(tl.int64)
    query_end = tl.load(query_start_ptr + request_idx + 1).to(tl.int64)
    query_base = query_start + query_block_idx * block_m
    if query_base >= query_end:
        return

    query_offsets = query_base + tl.arange(0, block_m)
    query_mask = query_offsets < query_end
    query_head_offset = head_idx.to(tl.int64) * head_dim
    key_lo, key_hi = _bounds(positions_ptr, query_offsets, query_mask, window)
    _, denominator, accumulator = _attend(
        q_ptr,
        kv_ptr,
        block_table_ptr,
        positions_ptr,
        query_offsets,
        query_mask,
        request_idx,
        key_lo,
        key_hi,
        q_stride,
        kv_stride,
        table_stride,
        softmax_scale,
        value_offset,
        window,
        head_dim,
        block_size,
        block_m,
        block_n,
        query_head_offset,
    )
    output = accumulator / denominator[:, None]
    dim_offsets = tl.arange(0, head_dim)
    tl.store(
        output_ptr
        + query_offsets[:, None] * output_stride
        + query_head_offset
        + dim_offsets[None, :],
        output.to(output_ptr.dtype.element_ty),
        mask=query_mask[:, None],
    )


@triton.jit
def _probe_attention_split_kernel(
    q_ptr,
    kv_ptr,
    max_ptr,
    sum_ptr,
    accumulator_ptr,
    block_table_ptr,
    positions_ptr,
    query_start_ptr,
    q_stride,
    kv_stride,
    table_stride,
    softmax_scale,
    stat_query_stride,
    stat_head_stride,
    accumulator_query_stride,
    accumulator_head_stride,
    accumulator_split_stride,
    num_splits: tl.constexpr,
    value_offset: tl.constexpr,
    window: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    split_request_idx = tl.program_id(0)
    request_idx = split_request_idx // num_splits
    split_idx = split_request_idx % num_splits
    head_idx = tl.program_id(1)
    query_block_idx = tl.program_id(2)
    query_start = tl.load(query_start_ptr + request_idx).to(tl.int64)
    query_end = tl.load(query_start_ptr + request_idx + 1).to(tl.int64)
    query_base = query_start + query_block_idx * block_m
    if query_base >= query_end:
        return

    query_offsets = query_base + tl.arange(0, block_m)
    query_mask = query_offsets < query_end
    query_head_offset = head_idx.to(tl.int64) * head_dim
    key_lo, key_hi = _bounds(positions_ptr, query_offsets, query_mask, window)
    keys_per_split = tl.cdiv(key_hi - key_lo, num_splits)
    split_lo = key_lo + split_idx * keys_per_split
    split_hi = tl.minimum(split_lo + keys_per_split, key_hi)
    running_max, denominator, accumulator = _attend(
        q_ptr,
        kv_ptr,
        block_table_ptr,
        positions_ptr,
        query_offsets,
        query_mask,
        request_idx,
        split_lo,
        split_hi,
        q_stride,
        kv_stride,
        table_stride,
        softmax_scale,
        value_offset,
        window,
        head_dim,
        block_size,
        block_m,
        block_n,
        query_head_offset,
    )
    stat_offset = (
        query_offsets * stat_query_stride + head_idx * stat_head_stride + split_idx
    )
    tl.store(max_ptr + stat_offset, running_max, mask=query_mask)
    tl.store(sum_ptr + stat_offset, denominator, mask=query_mask)
    dim_offsets = tl.arange(0, head_dim)
    tl.store(
        accumulator_ptr
        + query_offsets[:, None] * accumulator_query_stride
        + head_idx * accumulator_head_stride
        + split_idx * accumulator_split_stride
        + dim_offsets[None, :],
        accumulator,
        mask=query_mask[:, None],
    )


@triton.jit
def _probe_attention_combine_kernel(
    max_ptr,
    sum_ptr,
    accumulator_ptr,
    output_ptr,
    stat_query_stride,
    stat_head_stride,
    accumulator_query_stride,
    accumulator_head_stride,
    accumulator_split_stride,
    output_stride,
    num_splits: tl.constexpr,
    head_dim: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_offsets = tl.arange(0, num_splits)
    dim_offsets = tl.arange(0, head_dim)
    stat_offset = (
        query_idx * stat_query_stride + head_idx * stat_head_stride + split_offsets
    )
    maxima = tl.load(max_ptr + stat_offset)
    denominators = tl.load(sum_ptr + stat_offset)
    global_max = tl.max(maxima, 0)
    scales = tl.exp(maxima - global_max)
    accumulators = tl.load(
        accumulator_ptr
        + query_idx * accumulator_query_stride
        + head_idx * accumulator_head_stride
        + split_offsets[:, None] * accumulator_split_stride
        + dim_offsets[None, :]
    )
    output = tl.sum(accumulators * scales[:, None], 0) / tl.sum(
        denominators * scales, 0
    )
    tl.store(
        output_ptr
        + query_idx * output_stride
        + head_idx.to(tl.int64) * head_dim
        + dim_offsets,
        output.to(output_ptr.dtype.element_ty),
    )


@triton.jit
def _store_probe_kv_kernel(
    source_ptr,
    cache_ptr,
    slot_mapping_ptr,
    source_stride,
    cache_stride,
    width: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    slot = tl.load(slot_mapping_ptr + row)
    mask = (offsets < width) & (slot >= 0)
    values = tl.load(source_ptr + row * source_stride + offsets, mask=mask, other=0.0)
    tl.store(cache_ptr + slot * cache_stride + offsets, values, mask=mask)


def store_probe_kv(
    source: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    width = source.shape[1]
    _store_probe_kv_kernel[(source.shape[0],)](
        source,
        cache,
        slot_mapping,
        source.stride(0),
        cache.stride(0),
        width=width,
        block=triton.next_power_of_2(width),
    )


def _pick_splits(num_blocks: int, device: torch.device, max_kv: int) -> int:
    global _target_blocks
    if _target_blocks is None:
        properties = torch.cuda.get_device_properties(device)
        _target_blocks = 4 * properties.multi_processor_count
    tiles = max(1, max_kv // 64)
    split_cap = min(MAX_SPLITS, 1 << (tiles.bit_length() - 1))
    splits = 1
    while num_blocks * splits < _target_blocks and splits < split_cap:
        splits *= 2
    return splits


def probe_paged_attention(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    window: int | None,
    force_splits: int | None = None,
) -> torch.Tensor:
    total_queries = query.shape[0]
    output = torch.empty_like(query)
    num_requests = block_table.shape[0]
    if total_queries == 0 or num_requests == 0:
        return output
    block_m = 16 if max_query_len <= 16 else 64
    query_blocks = triton.cdiv(max_query_len, block_m)
    splits = force_splits
    if splits is None:
        splits = _pick_splits(
            num_requests * num_heads * query_blocks,
            query.device,
            window or block_table.shape[1] * block_size,
        )

    launch_kwargs = {
        "value_offset": head_dim,
        "window": window or 0,
        "head_dim": head_dim,
        "block_size": block_size,
        "block_m": block_m,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    }
    if splits == 1:
        _probe_attention_kernel[(num_requests, num_heads, query_blocks)](
            query,
            kv_cache,
            output,
            block_table,
            positions,
            query_start_loc,
            query.stride(0),
            kv_cache.stride(0),
            output.stride(0),
            block_table.stride(0),
            1.0 / math.sqrt(head_dim),
            **launch_kwargs,
        )
        return output

    maxima = torch.empty(
        total_queries, num_heads, splits, device=query.device, dtype=torch.float32
    )
    denominators = torch.empty_like(maxima)
    accumulators = torch.empty(
        total_queries,
        num_heads,
        splits,
        head_dim,
        device=query.device,
        dtype=torch.float32,
    )
    _probe_attention_split_kernel[(num_requests * splits, num_heads, query_blocks)](
        query,
        kv_cache,
        maxima,
        denominators,
        accumulators,
        block_table,
        positions,
        query_start_loc,
        query.stride(0),
        kv_cache.stride(0),
        block_table.stride(0),
        1.0 / math.sqrt(head_dim),
        maxima.stride(0),
        maxima.stride(1),
        accumulators.stride(0),
        accumulators.stride(1),
        accumulators.stride(2),
        num_splits=splits,
        **launch_kwargs,
    )
    _probe_attention_combine_kernel[(total_queries, num_heads)](
        maxima,
        denominators,
        accumulators,
        output,
        maxima.stride(0),
        maxima.stride(1),
        accumulators.stride(0),
        accumulators.stride(1),
        accumulators.stride(2),
        output.stride(0),
        num_splits=splits,
        head_dim=head_dim,
        num_warps=4,
    )
    return output
