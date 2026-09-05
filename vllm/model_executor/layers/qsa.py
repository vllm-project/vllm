# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4 QSA kernels for vLLM.

Self-developed Triton kernels for Qwen4's QSA (Query-Sparse Attention) mechanism,
including MQA paged attention with compressed key cache, KV cache storage, and
group-wise compression with multiaxis RoPE.
"""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton


def _is_triton_device(*tensors: torch.Tensor) -> bool:
    """Return true for same-device accelerator tensors accepted by Triton."""

    return bool(
        tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.device.type not in ("cpu", "meta") for tensor in tensors)
    )


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _qsa_mqa_paged_dot_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    """Tensor-core/MFMA QSA score path expressed with portable ``tl.dot``."""

    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[:, None] * stride_q_head
        + dims[None, :] * stride_q_dim,
        mask=(heads[:, None] < NUM_HEADS) & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    keys = tl.load(
        k_cache_ptr
        + safe_page[None, :] * stride_cache_block
        + page_offset[None, :] * stride_cache_token
        + dims[:, None] * stride_cache_dim,
        mask=(dims[:, None] < HEAD_DIM) & valid[None, :],
        other=0.0,
    )
    dots = tl.dot(query, keys)
    score = tl.sum(tl.maximum(dots, 0.0), axis=0) / score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _store_qsa_kv_rows_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_rows_ptr,
    v_rows_ptr,
    stride_k_cache_block,
    stride_k_cache_token,
    stride_k_cache_head,
    stride_k_cache_dim,
    stride_v_cache_block,
    stride_v_cache_token,
    stride_v_cache_head,
    stride_v_cache_dim,
    stride_k_rows_row,
    stride_k_rows_head,
    stride_k_rows_dim,
    stride_v_rows_row,
    stride_v_rows_head,
    stride_v_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Store K and V together while preserving arbitrary cache strides."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    k_values = tl.load(
        k_rows_ptr
        + row * stride_k_rows_row
        + head * stride_k_rows_head
        + dims * stride_k_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    v_values = tl.load(
        v_rows_ptr
        + row * stride_v_rows_row
        + head * stride_v_rows_head
        + dims * stride_v_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    tl.store(
        k_cache_ptr
        + block * stride_k_cache_block
        + token * stride_k_cache_token
        + head * stride_k_cache_head
        + dims * stride_k_cache_dim,
        k_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )
    tl.store(
        v_cache_ptr
        + block * stride_v_cache_block
        + token * stride_v_cache_token
        + head * stride_v_cache_head
        + dims * stride_v_cache_dim,
        v_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    rope_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_raw_table_req,
    stride_raw_table_page,
    stride_rope_table_req,
    stride_rope_table_page,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_raw_requests,
    num_rope_requests,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    ROPE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_raw_requests)
        & (request < num_rope_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_raw_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_raw_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            values = tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            accumulator += values

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        rope_logical_page = first_position // ROPE_PAGE_SIZE
        rope_page_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_logical_page < ROPE_TABLE_WIDTH)
        rope_physical_page = tl.load(
            rope_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_rope_requests - 1)
            * stride_rope_table_req
            + tl.minimum(rope_logical_page, ROPE_TABLE_WIDTH - 1)
            * stride_rope_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (rope_physical_page < num_rope_blocks)
        rope_values = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_page_offset * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_rope & (position_dims < 3),
            other=0,
        )
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            rope_values,
            mask=(row < num_rows) & (position_dims < 3),
        )
    else:
        first_position = tl.where(valid_row, first_position, 0)
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            first_position,
            mask=(row < num_rows) & (position_dims < 3),
        )


@triton.jit
def _compress_norm_mrope_store_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    norm_weight_ptr,
    cos_sin_cache_ptr,
    compressed_cache_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_table_req,
    stride_table_page,
    stride_cos_row,
    stride_cos_dim,
    stride_compressed_block,
    stride_compressed_token,
    stride_compressed_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_compressed_blocks,
    num_requests,
    num_cos_rows,
    norm_eps,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MROPE_SECTION_T: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    LOAD_MROPE_POSITIONS: tl.constexpr,
) -> None:
    """Pool, Gemma-normalize, MRoPE, and store one compressed QSA key."""

    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
        & (compressed_slot < num_compressed_blocks * COMPRESSED_PAGE_SIZE)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            accumulator += tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    # Match the unfused BF16 materialization before Gemma RMSNorm.
    pooled = (accumulator / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_fp32 = pooled.to(tl.float32)
    variance = tl.sum(pooled_fp32 * pooled_fp32, axis=0) / HEAD_DIM
    weight = tl.load(
        norm_weight_ptr + dims,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    normalized = (pooled_fp32 * tl.rsqrt(variance + norm_eps) * (weight + 1.0)).to(
        tl.bfloat16
    )

    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_MROPE_POSITIONS:
        rope_page = first_position // ROPE_PAGE_SIZE
        rope_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_page < RAW_TABLE_WIDTH)
        rope_physical_page = tl.load(
            raw_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1) * stride_table_req
            + tl.minimum(rope_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (rope_physical_page < num_rope_blocks)
        axis_offsets = tl.arange(0, 4)
        axis_positions = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_offset * stride_rope_token
            + axis_offsets * stride_rope_dim,
            mask=valid_rope & (axis_offsets < 3),
            other=0,
        )
        time_position = tl.max(tl.where(axis_offsets == 0, axis_positions, 0))
        height_position = tl.max(tl.where(axis_offsets == 1, axis_positions, 0))
        width_position = tl.max(tl.where(axis_offsets == 2, axis_positions, 0))
    else:
        time_position = first_position
        height_position = first_position
        width_position = first_position

    # HEAD_DIM=128 and ROTARY_DIM=64 in the production checkpoint.  Split the
    # normalized vector without dynamic local indexing: first head half holds
    # all rotary channels, second head half is the pass-through tail.
    head_pairs = tl.permute(tl.reshape(normalized, (2, BLOCK_D // 2)), (1, 0))
    rotary_values, pass_values = tl.split(head_pairs)
    rotary_pairs = tl.permute(tl.reshape(rotary_values, (2, ROTARY_DIM // 2)), (1, 0))
    first_half, second_half = tl.split(rotary_pairs)
    frequencies = tl.arange(0, ROTARY_DIM // 2)
    if MROPE_INTERLEAVED:
        use_height = ((frequencies % 3) == 1) & (frequencies < 3 * MROPE_SECTION_H)
        use_width = ((frequencies % 3) == 2) & (frequencies < 3 * MROPE_SECTION_W)
    else:
        height_start = MROPE_SECTION_T
        width_start = height_start + MROPE_SECTION_H
        use_height = (frequencies >= height_start) & (frequencies < width_start)
        use_width = (frequencies >= width_start) & (
            frequencies < width_start + MROPE_SECTION_W
        )
    rope_positions = tl.where(
        use_height,
        height_position,
        tl.where(use_width, width_position, time_position),
    )
    valid_position = (rope_positions >= 0) & (rope_positions < num_cos_rows)
    safe_positions = tl.minimum(tl.maximum(rope_positions, 0), num_cos_rows - 1)
    cos = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + frequencies * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + (ROTARY_DIM // 2 + frequencies) * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    rotated_first = (first_half * cos - second_half * sin).to(tl.bfloat16)
    rotated_second = (second_half * cos + first_half * sin).to(tl.bfloat16)

    compressed_block = tl.maximum(compressed_slot, 0) // COMPRESSED_PAGE_SIZE
    compressed_token = tl.maximum(compressed_slot, 0) % COMPRESSED_PAGE_SIZE
    compressed_base = (
        compressed_cache_ptr
        + compressed_block.to(tl.int64) * stride_compressed_block
        + compressed_token * stride_compressed_token
    )
    tl.store(
        compressed_base + frequencies * stride_compressed_dim,
        rotated_first,
        mask=valid_row & valid_position,
    )
    tl.store(
        compressed_base + (ROTARY_DIM // 2 + frequencies) * stride_compressed_dim,
        rotated_second,
        mask=valid_row & valid_position,
    )
    pass_offsets = tl.arange(0, BLOCK_D // 2)
    tl.store(
        compressed_base + (ROTARY_DIM + pass_offsets) * stride_compressed_dim,
        pass_values,
        mask=valid_row & ((ROTARY_DIM + pass_offsets) < HEAD_DIM),
    )


def qwen4_qsa_mqa_paged_dot(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int = 4,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA indexer scores directly from a paged compressed-key cache.

    This entry point intentionally selects the self-developed ``tl.dot``
    kernel.  The Qwen4 production contract is BF16, four query heads, and
    head dimension 128; callers with another shape must use a different
    implementation instead of silently falling back to Torch.
    """

    if not _is_triton_device(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths
    ):
        raise RuntimeError("Qwen4 QSA MQA dot requires a Triton accelerator")
    if q.ndim != 3 or q.shape[1:] != (4, 128) or q.dtype != torch.bfloat16:
        raise ValueError("Qwen4 QSA MQA dot requires BF16 q shaped [rows, 4, 128]")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("Qwen4 QSA MQA cache must be [pages, page_size, 1, 128]")
    if k_cache.dtype != q.dtype:
        raise ValueError("Qwen4 QSA query and cache must have the same dtype")
    if page_table.ndim != 2:
        raise ValueError("Qwen4 QSA MQA page table must be rank-2")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA page table must use int32 or int64")
    rows = q.shape[0]
    if rows and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError(
            "Qwen4 QSA MQA cache and page table must be nonempty for nonempty q"
        )
    if token_to_req.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("Qwen4 QSA request metadata must match query rows")
    if token_to_req.dtype not in (
        torch.int32,
        torch.int64,
    ) or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Qwen4 QSA request metadata must use int32 or int64")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("Qwen4 QSA sequence lengths must match page-table requests")
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA sequence lengths must use int32 or int64")
    if compress_ratio <= 0:
        raise ValueError("Qwen4 QSA compression ratio must be positive")
    divisor = math.sqrt(128) if score_scale is None else float(score_scale)
    if divisor <= 0:
        raise ValueError("Qwen4 QSA score scale must be positive")
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else int(num_columns)
    if columns < 0 or columns > capacity:
        raise ValueError("Qwen4 QSA score width must be in [0, page-table capacity]")
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty((rows,), dtype=torch.int32, device=q.device)
    if rows == 0 or columns == 0:
        return logits, visible_blocks
    block_n = 32
    _qsa_mqa_paged_dot_kernel[(rows, triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        rows,
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        divisor,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_H=16,
        BLOCK_N=block_n,
        BLOCK_D=128,
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
        num_stages=2,
    )
    return logits, visible_blocks



def qwen4_store_qsa_kv_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store paired QSA K/V rows in one stride-aware Triton launch.

    The production fast-path precondition is that valid slots are unique.  An
    invalid slot is safely ignored by the kernel; duplicate slots are outside
    the parallel write contract and must be removed by the caller.
    """

    if not _is_triton_device(k_cache, v_cache, slot_mapping, key, value):
        raise RuntimeError("Qwen4 QSA K/V store requires a Triton accelerator")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("Qwen4 QSA K/V caches must be [blocks, page, heads, dim]")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("Qwen4 QSA K/V rows must be [rows, heads, dim]")
    if key.shape != (slot_mapping.numel(), k_cache.shape[2], k_cache.shape[3]):
        raise ValueError("Qwen4 QSA K/V rows and slot mapping have incompatible shapes")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA slot mapping must use int32 or int64")
    if key.dtype != k_cache.dtype or value.dtype != v_cache.dtype:
        raise ValueError("Qwen4 QSA K/V rows and caches must have matching dtypes")
    if not all(k_cache.shape):
        raise ValueError("Qwen4 QSA K/V caches must be nonempty")
    if not key.shape[0]:
        return
    _store_qsa_kv_rows_kernel[(key.shape[0], key.shape[1])](
        k_cache,
        v_cache,
        slot_mapping,
        key,
        value,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.shape[0],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=key.shape[1],
        HEAD_DIM=key.shape[2],
        BLOCK_D=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )



def qwen4_compress_norm_mrope_store_groups(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int = 4,
    norm_eps: float = 1.0e-6,
    rotary_dim: int = 64,
    mrope_section: tuple[int, int, int] = (11, 11, 10),
    mrope_interleaved: bool = True,
    rope_cache: torch.Tensor | None = None,
) -> None:
    """Fuse QSA group pooling, Gemma RMSNorm, interleaved MRoPE, and store.

    For the Qwen4 cache, ``mrope_section=(11, 11, 10)`` describes counts of
    frequency lanes.  With ``mrope_interleaved=True`` the actual lane selector
    is ``frequency % 3`` (T/H/W), not three contiguous slices.
    """

    tensors = [
        raw_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        compressed_cache,
        norm_weight,
        cos_sin_cache,
    ]
    if rope_cache is not None:
        tensors.append(rope_cache)
    if not _is_triton_device(*tensors):
        raise RuntimeError("Qwen4 QSA compression requires a Triton accelerator")
    if (
        raw_cache.ndim != 4
        or raw_cache.shape[2:] != (1, 128)
        or not all(raw_cache.shape)
    ):
        raise ValueError("Qwen4 QSA raw cache must be nonempty [blocks, page, 1, 128]")
    if (
        compressed_cache.ndim != 4
        or compressed_cache.shape[2:] != (1, 128)
        or not all(compressed_cache.shape)
    ):
        raise ValueError("Qwen4 QSA compressed cache must be [blocks, page, 1, 128]")
    if compressed_cache.dtype != raw_cache.dtype:
        raise ValueError("Qwen4 QSA raw and compressed caches must match dtype")
    if raw_block_table.ndim != 2:
        raise ValueError("Qwen4 QSA raw block table must be rank-2")
    if raw_block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA raw block table must use int32 or int64")
    rows = token_to_req.numel()
    if rows and not all(raw_block_table.shape):
        raise ValueError("Qwen4 QSA raw block table must be nonempty for nonempty rows")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("Qwen4 QSA compression metadata must match token rows")
    if (
        token_to_req.dtype not in (torch.int32, torch.int64)
        or logical_positions.dtype
        not in (
            torch.int32,
            torch.int64,
        )
        or compressed_slots.dtype not in (torch.int32, torch.int64)
    ):
        raise TypeError("Qwen4 QSA compression metadata must use int32 or int64")
    if norm_weight.shape != (128,) or norm_weight.dtype != raw_cache.dtype:
        raise ValueError("Qwen4 QSA norm weight must be a same-dtype [128] vector")
    if norm_weight.stride(0) != 1:
        raise ValueError("Qwen4 QSA norm weight must be contiguous")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("Qwen4 QSA cos/sin cache must be [positions, rotary_dim]")
    if cos_sin_cache.dtype != raw_cache.dtype or not cos_sin_cache.shape[0]:
        raise ValueError(
            "Qwen4 QSA cos/sin cache must be nonempty and match cache dtype"
        )
    if (
        len(mrope_section) != 3
        or any(section < 0 for section in mrope_section)
        or rotary_dim != 64
        or sum(mrope_section) != rotary_dim // 2
    ):
        raise ValueError(
            "Qwen4 QSA MRoPE requires rotary_dim=64 and sections summing to 32"
        )
    if compress_ratio <= 0 or norm_eps <= 0:
        raise ValueError(
            "Qwen4 QSA compression ratio and norm epsilon must be positive"
        )
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError(
            "Qwen4 QSA packed MRoPE cache must be [blocks, page, 1, 3] int64"
        )
    if rows == 0:
        return
    if rope_cache is None:
        rope_cache = raw_cache
        load_mrope_positions = False
    else:
        load_mrope_positions = True
    _compress_norm_mrope_store_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        norm_weight,
        cos_sin_cache,
        compressed_cache,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(3),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        compressed_cache.shape[0],
        raw_block_table.shape[0],
        cos_sin_cache.shape[0],
        float(norm_eps),
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        COMPRESSED_PAGE_SIZE=compressed_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=128,
        ROTARY_DIM=rotary_dim,
        BLOCK_D=128,
        MROPE_SECTION_T=mrope_section[0],
        MROPE_SECTION_H=mrope_section[1],
        MROPE_SECTION_W=mrope_section[2],
        MROPE_INTERLEAVED=mrope_interleaved,
        LOAD_MROPE_POSITIONS=load_mrope_positions,
        num_warps=4,
    )


__all__ = [
    "_compress_qsa_groups_kernel",
    "_compress_norm_mrope_store_qsa_groups_kernel",
    "_qsa_mqa_paged_kernel",
    "_qsa_mqa_paged_dot_kernel",
    "_store_qsa_rows_kernel",
    "_store_qsa_kv_rows_kernel",
    "qwen4_compress_norm_mrope_store_groups",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_store_qsa_kv_rows",
]
