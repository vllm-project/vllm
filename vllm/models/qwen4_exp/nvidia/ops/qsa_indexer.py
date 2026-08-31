# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA index selection."""

import math

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_LARGE_DECODE_REQUESTS = 33


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_columns,
    num_pages,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    DECODE_QUERY_LEN_PADDED: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    MMA_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    positions = tl.load(
        query_positions_ptr + rows,
        mask=valid_query_offsets,
        other=-1,
    )
    sequence_length = tl.load(sequence_lengths_ptr + request)
    visible = tl.maximum(
        0,
        tl.minimum(
            (positions + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
    ).to(tl.int32)
    visible = tl.where(valid_query_offsets, visible, 0)
    if tl.program_id(1) == 0:
        tl.store(
            visible_blocks_ptr + rows,
            visible,
            mask=valid_query_offsets,
        )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, MMA_N)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2) / score_divisor
        valid = (columns[:, None] < visible[None, :]) & page_valid[:, None]
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            tl.where(valid, score, -float("inf")),
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < num_columns)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_mqa_paged_prefill_kernel(
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
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * TILE_R
    lanes = tl.arange(0, TILE_R)
    rows = row_start + lanes
    valid_rows = rows < num_rows
    req_vec = tl.load(token_to_req_ptr + rows, mask=valid_rows, other=-1)
    valid_reqs = valid_rows & (req_vec >= 0) & (req_vec < num_requests)
    safe_req_vec = tl.minimum(tl.maximum(req_vec, 0), num_requests - 1)
    seq_vec = tl.load(
        sequence_lengths_ptr + safe_req_vec,
        mask=valid_reqs,
        other=0,
    )
    positions = tl.load(query_positions_ptr + rows, mask=valid_rows, other=-1)
    visible = tl.maximum(
        0,
        tl.minimum(
            (positions + 1) // COMPRESS_RATIO,
            seq_vec // COMPRESS_RATIO,
        ),
    )
    visible = tl.where(valid_reqs, visible, 0).to(tl.int32)
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=valid_rows)
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(1) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_rows = row_start + m // NUM_HEADS_PADDED
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (q_rows[None, :] < num_rows)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)

    last_row = tl.minimum(row_start + TILE_R, num_rows) - 1
    first_req = tl.load(token_to_req_ptr + row_start)
    last_req = tl.load(token_to_req_ptr + last_row)
    if first_req == last_req:
        for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
            columns = tile * BLOCK_N + column_offsets
            live = columns < max_visible
            logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
            page_offset = columns % PAGE_SIZE
            physical_page = tl.load(
                page_table_ptr + first_req * stride_table_req + logical_page,
                mask=live,
                other=-1,
            )
            page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
            safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
            keys = tl.load(
                k_cache_ptr
                + safe_physical_page[:, None] * stride_cache_block
                + page_offset[:, None] * stride_cache_token
                + dims[None, :],
                mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
                other=0.0,
                eviction_policy="evict_first",
            )
            scores = tl.dot(keys, query, out_dtype=tl.float32)
            scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
            score = tl.sum(tl.maximum(scores, 0.0), axis=2) / score_divisor
            store_mask = (
                valid_reqs[None, :]
                & (columns[:, None] < visible[None, :])
                & (columns[:, None] < num_columns)
            )
            tl.store(
                logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
                tl.where(page_valid[:, None], score, -float("inf")),
                mask=store_mask,
            )
    else:
        segment_start = tl.min(lanes, axis=0)
        request = tl.load(token_to_req_ptr + row_start)
        while (segment_start < TILE_R) & (request >= 0):
            segment_end = tl.min(
                tl.where(
                    (lanes > segment_start) & (req_vec != request),
                    lanes,
                    TILE_R,
                ),
                axis=0,
            )
            segment_rows = (lanes >= segment_start) & (lanes < segment_end) & valid_rows
            segment_visible = tl.max(tl.where(segment_rows, visible, 0), axis=0)
            segment_tile_end = tl.minimum(k_tile_end, tl.cdiv(segment_visible, BLOCK_N))
            for tile in tl.range(k_tile_start, segment_tile_end):
                columns = tile * BLOCK_N + column_offsets
                live = columns < segment_visible
                logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
                page_offset = columns % PAGE_SIZE
                physical_page = tl.load(
                    page_table_ptr + request * stride_table_req + logical_page,
                    mask=live,
                    other=-1,
                )
                page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
                safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
                keys = tl.load(
                    k_cache_ptr
                    + safe_physical_page[:, None] * stride_cache_block
                    + page_offset[:, None] * stride_cache_token
                    + dims[None, :],
                    mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
                    other=0.0,
                    eviction_policy="evict_first",
                )
                scores = tl.dot(keys, query, out_dtype=tl.float32)
                scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
                score = tl.sum(tl.maximum(scores, 0.0), axis=2) / score_divisor
                store_mask = (
                    segment_rows[None, :]
                    & (columns[:, None] < visible[None, :])
                    & (columns[:, None] < num_columns)
                )
                tl.store(
                    logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
                    tl.where(page_valid[:, None], score, -float("inf")),
                    mask=store_mask,
                )
            segment_start = segment_end
            request = tl.load(
                token_to_req_ptr + row_start + segment_start,
                mask=(segment_start < TILE_R) & (row_start + segment_start < num_rows),
                other=-1,
            )


@triton.jit(do_not_specialize=["rows", "num_requests"])
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    profiles = []
    for dql in range(1, max_dql + 1):
        if dql <= max_num_batched_tokens:
            profiles.append((dql, 1))
        if (
            max_num_reqs >= _LARGE_DECODE_REQUESTS
            and dql * _LARGE_DECODE_REQUESTS <= max_num_batched_tokens
        ):
            profiles.append((dql, _LARGE_DECODE_REQUESTS))
    return tuple(profiles)


def _qsa_decode_kernel_config(
    *,
    page_size: int,
    page_table_width: int,
    num_heads: int,
    head_dim: int,
    decode_query_len: int,
    num_requests: int,
    compress_ratio: int,
) -> dict[str, int]:
    decode_query_len_padded = triton.next_power_of_2(decode_query_len)
    num_heads_padded = triton.next_power_of_2(num_heads)
    return {
        "PAGE_SIZE": page_size,
        "PAGE_TABLE_WIDTH": page_table_width,
        "NUM_HEADS": num_heads,
        "HEAD_DIM": head_dim,
        "DECODE_QUERY_LEN": decode_query_len,
        "DECODE_QUERY_LEN_PADDED": decode_query_len_padded,
        "NUM_HEADS_PADDED": num_heads_padded,
        "MMA_N": decode_query_len_padded * num_heads_padded,
        "BLOCK_N": 64,
        "BLOCK_D": max(16, triton.next_power_of_2(head_dim)),
        "TILES_PER_PROG": 1 if num_requests <= 32 else 8,
        "STAGES": 2,
        "COMPRESS_RATIO": compress_ratio,
    }


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    compress_ratio: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""

    if not current_platform.is_cuda() or not HAS_TRITON:
        return ()
    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
    )
    if not profiles:
        return ()

    warmup = getattr(_qsa_mqa_paged_uniform_kernel, "warmup", None)
    assert warmup is not None
    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    int32_ptr = TritonWarmupTensor(torch.int32)
    int64_ptr = TritonWarmupTensor(torch.int64)

    for decode_query_len, num_requests in profiles:
        num_rows = decode_query_len * num_requests
        q_ptr = TritonWarmupTensor(
            torch.bfloat16,
            shape=(num_rows, num_heads, head_dim),
        )
        logits_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_rows, columns),
        )
        kernel_config = _qsa_decode_kernel_config(
            page_size=page_size,
            page_table_width=page_table_width,
            num_heads=num_heads,
            head_dim=head_dim,
            decode_query_len=decode_query_len,
            num_requests=num_requests,
            compress_ratio=compress_ratio,
        )
        warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            int64_ptr,
            int32_ptr,
            int32_ptr,
            logits_ptr,
            q_ptr.stride()[0],
            q_ptr.stride()[1],
            k_cache_ptr.stride()[0],
            k_cache_ptr.stride()[1],
            page_table_ptr.stride()[0],
            logits_ptr.stride()[0],
            columns,
            k_cache.shape[0],
            math.sqrt(head_dim),
            **kernel_config,
            num_warps=2,
            grid=(
                num_requests,
                triton.cdiv(
                    columns,
                    kernel_config["BLOCK_N"] * kernel_config["TILES_PER_PROG"],
                ),
            ),
        )
    return profiles


def qsa_mqa_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    decode_query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a request-major uniform decode or speculative-decode batch."""

    _validate_mqa(q)
    if decode_query_len <= 0:
        raise ValueError("QSA decode query length must be positive")
    if q.shape[0] % decode_query_len:
        raise ValueError("QSA decode rows must be divisible by the query length")
    num_requests = q.shape[0] // decode_query_len
    if page_table.ndim != 2 or page_table.shape[0] != num_requests:
        raise ValueError("QSA decode page table must have one row per request")
    if sequence_lengths.shape != (num_requests,):
        raise ValueError("QSA decode sequence lengths must match requests")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA decode positions must match query rows")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2] or compress_ratio <= 0:
        raise ValueError("QSA decode scoring received incompatible dimensions")
    if q.stride(2) != 1:
        raise ValueError("QSA decode query head dimension must be contiguous")
    if k_cache.stride(3) != 1:
        raise ValueError("QSA decode cache head dimension must be contiguous")
    if page_table.stride(1) != 1:
        raise ValueError("QSA decode page-table rows must be contiguous")
    if query_positions.stride(0) != 1 or sequence_lengths.stride(0) != 1:
        raise ValueError("QSA decode row metadata must be contiguous")
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA decode scoring requires CUDA and Triton")

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    block_n = 64
    tiles_per_program = 1 if num_requests <= 32 else 8
    stages = 2
    decode_query_len_padded = triton.next_power_of_2(decode_query_len)
    num_heads_padded = triton.next_power_of_2(q.shape[1])
    grid = (
        num_requests,
        triton.cdiv(columns, block_n * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        columns,
        k_cache.shape[0],
        math.sqrt(q.shape[2]),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        DECODE_QUERY_LEN_PADDED=decode_query_len_padded,
        NUM_HEADS_PADDED=num_heads_padded,
        MMA_N=decode_query_len_padded * num_heads_padded,
        BLOCK_N=block_n,
        BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
        TILES_PER_PROG=tiles_per_program,
        STAGES=stages,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


def qsa_mqa_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a potentially multi-request prefill workspace chunk."""

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA prefill scoring requires CUDA and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2] or compress_ratio <= 0:
        raise ValueError("QSA prefill scoring received incompatible dimensions")
    if page_table.ndim != 2 or sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA prefill request metadata has incompatible shapes")
    if token_to_req.shape != query_positions.shape or token_to_req.shape != (
        q.shape[0],
    ):
        raise ValueError("QSA prefill row metadata must match query rows")
    if q.stride(2) != 1:
        raise ValueError("QSA prefill query head dimension must be contiguous")
    if k_cache.stride(3) != 1:
        raise ValueError("QSA prefill cache head dimension must be contiguous")
    if page_table.stride(1) != 1:
        raise ValueError("QSA prefill page-table rows must be contiguous")
    if (
        token_to_req.stride(0) != 1
        or query_positions.stride(0) != 1
        or sequence_lengths.stride(0) != 1
    ):
        raise ValueError("QSA prefill row metadata must be contiguous")

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    tile_r = 32
    block_n = 128
    k_tiles = 8
    grid = (
        triton.cdiv(q.shape[0], tile_r),
        triton.cdiv(columns, block_n * k_tiles),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
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
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        math.sqrt(q.shape[2]),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        NUM_HEADS_PADDED=triton.next_power_of_2(q.shape[1]),
        HEAD_DIM=q.shape[2],
        BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
        TILE_R=tile_r,
        BLOCK_N=block_n,
        K_TILES=k_tiles,
        COMPRESS_RATIO=compress_ratio,
        STAGES=2,
        num_warps=4,
    )
    return logits, visible_blocks


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the causal tail of the open group."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA expansion requires Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def _selection_output(
    q: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None,
) -> torch.Tensor:
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    return out


def _selection_workspace(
    rows: int,
    token_topk: int,
    compress_ratio: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    blocks_buffer = torch.empty((rows, block_topk), dtype=torch.int32, device=device)
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
    )
    return blocks_buffer, topk_workspace


def _select_and_expand_qsa_scores(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor,
    blocks: torch.Tensor,
    topk_workspace: torch.Tensor,
) -> None:
    block_topk = token_topk // compress_ratio
    use_cooperative_topk = (
        logits.shape[0] <= 32
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        blocks,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )
    expand_qsa_block_indices_cuda(
        blocks,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score and select a homogeneous request-major decode batch.

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        token_to_req: Request index for each query, shaped ``[num_queries]``.
        query_positions: Logical position per query, shaped ``[num_queries]``.
        sequence_lengths: Sequence length per request, shaped ``[num_requests]``.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        out: Optional output buffer shaped ``[num_queries,
            token_topk + compress_ratio - 1]``.

    Returns:
        Request-relative logical token indices shaped ``[num_queries,
        token_topk + compress_ratio - 1]``.
    """

    out = _selection_output(q, token_topk, compress_ratio, out)
    if not q.shape[0]:
        return out
    blocks, topk_workspace = _selection_workspace(
        q.shape[0], token_topk, compress_ratio, q.device
    )
    logits, visible_blocks = qsa_mqa_paged_decode(
        q,
        k_cache,
        page_table,
        query_positions,
        sequence_lengths,
        compress_ratio,
        decode_query_len,
    )
    _select_and_expand_qsa_scores(
        logits,
        visible_blocks,
        query_positions,
        sequence_lengths,
        token_to_req,
        token_topk,
        compress_ratio,
        out,
        blocks,
        topk_workspace,
    )
    return out


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score and select a prefill batch in bounded workspace chunks.

    Args:
        q: Query tensor shaped ``[num_prefill_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        token_to_req: Request index for each query, shaped
            ``[num_prefill_tokens]``. Indices refer to rows in ``page_table``.
        query_positions: Logical position per query, shaped
            ``[num_prefill_tokens]``.
        sequence_lengths: Sequence length per request, shaped ``[num_requests]``.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        out: Optional output buffer shaped ``[num_prefill_tokens,
            token_topk + compress_ratio - 1]``.

    Returns:
        Request-relative logical token indices shaped ``[num_prefill_tokens,
        token_topk + compress_ratio - 1]``.
    """

    out = _selection_output(q, token_topk, compress_ratio, out)
    rows = q.shape[0]
    if not rows:
        return out
    columns = page_table.shape[1] * k_cache.shape[1]
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    work_rows = min(rows, rows_per_chunk)
    blocks_buffer, topk_workspace = _selection_workspace(
        work_rows, token_topk, compress_ratio, q.device
    )

    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged_prefill(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            query_positions[row_slice],
            sequence_lengths,
            compress_ratio,
        )
        chunk_rows = row_end - row_start
        _select_and_expand_qsa_scores(
            logits,
            visible_blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            token_topk,
            compress_ratio,
            out[row_slice],
            blocks_buffer[:chunk_rows],
            topk_workspace,
        )
    return out


__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_mqa_paged_decode",
    "qsa_mqa_paged_prefill",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_paged_decode",
]
