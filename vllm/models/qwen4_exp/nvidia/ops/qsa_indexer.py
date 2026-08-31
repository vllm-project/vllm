# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA index selection."""

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_DECODE_TILE_GROUPING_MIN_REQUESTS = 33


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    DECODE_QUERY_LEN_PADDED: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
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
        score = tl.sum(scores, axis=2) / HEAD_DIM**0.5
        valid = (columns[:, None] < visible[None, :]) & page_valid[:, None]
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            tl.where(valid, score, -float("inf")),
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < num_columns)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    num_pages,
    query_offset,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
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
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2) / HEAD_DIM**0.5
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            tl.where(page_valid[:, None], score, -float("inf")),
            mask=store_mask,
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    output_width: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
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
    valid = (columns < output_width) & (is_expanded | is_tail) & (token >= 0)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=columns < output_width,
    )


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
            max_num_reqs >= _DECODE_TILE_GROUPING_MIN_REQUESTS
            and dql * _DECODE_TILE_GROUPING_MIN_REQUESTS <= max_num_batched_tokens
        ):
            profiles.append((dql, _DECODE_TILE_GROUPING_MIN_REQUESTS))
    return tuple(profiles)


def _qsa_decode_kernel_config(
    *,
    page_size: int,
    page_table_width: int,
    num_heads: int,
    head_dim: int,
    decode_query_len: int,
    num_requests: int,
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
        "BLOCK_N": 64,
        "BLOCK_D": max(16, triton.next_power_of_2(head_dim)),
        "TILES_PER_PROG": (
            1 if num_requests < _DECODE_TILE_GROUPING_MIN_REQUESTS else 8
        ),
        "STAGES": 2,
    }


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""

    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
    )
    if not profiles:
        return ()

    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)

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
        )
        _qsa_mqa_paged_uniform_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            q_ptr.stride()[0],
            q_ptr.stride()[1],
            k_cache_ptr.stride()[0],
            k_cache_ptr.stride()[1],
            page_table_ptr.stride()[0],
            logits_ptr.stride()[0],
            k_cache.shape[0],
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
    visible_blocks: torch.Tensor,
    decode_query_len: int,
) -> torch.Tensor:
    """Score a request-major uniform decode or speculative-decode batch."""

    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    num_requests = q.shape[0] // decode_query_len
    assert page_table.shape[0] == num_requests
    assert visible_blocks.shape == (q.shape[0],)
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    if not q.shape[0] or not columns:
        return logits
    kernel_config = _qsa_decode_kernel_config(
        page_size=k_cache.shape[1],
        page_table_width=page_table.shape[1],
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        decode_query_len=decode_query_len,
        num_requests=num_requests,
    )
    grid = (
        num_requests,
        triton.cdiv(
            columns,
            kernel_config["BLOCK_N"] * kernel_config["TILES_PER_PROG"],
        ),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        k_cache.shape[0],
        **kernel_config,
        num_warps=2,
    )
    return logits


def _launch_qsa_mqa_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((num_queries, columns), dtype=torch.float32, device=q.device)
    if not num_queries or not columns:
        return logits
    tile_r = 32
    block_n = 128
    k_tiles = 8
    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), tile_r),
        triton.cdiv(columns, block_n * k_tiles),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        num_queries,
        k_cache.shape[0],
        query_offset,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        NUM_HEADS_PADDED=triton.next_power_of_2(q.shape[1]),
        HEAD_DIM=q.shape[2],
        BLOCK_D=max(16, triton.next_power_of_2(q.shape[2])),
        TILE_R=tile_r,
        BLOCK_N=block_n,
        K_TILES=k_tiles,
        STAGES=2,
        num_warps=4,
    )
    return logits


def qsa_mqa_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
) -> torch.Tensor:
    """Score a packed all-prefill batch."""

    return _launch_qsa_mqa_paged_prefill(
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        max_query_len,
        query_offset=0,
        num_queries=q.shape[0],
    )


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Expand compressed blocks and compact the causal tail of the open group."""

    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (query_positions.numel(), block_topk)
    assert visible_blocks.shape == query_positions.shape
    assert out.shape == (block_indices.shape[0], output_width)
    if not block_indices.shape[0]:
        return
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )


def _selection_workspace(device: torch.device) -> torch.Tensor:
    return torch.empty((_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=device)


def _select_qsa_scores(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    block_indices: torch.Tensor,
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
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score and select compressed blocks for a request-major decode batch.

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        block_indices: Compressed-index output buffer.
    """

    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    if not q.shape[0]:
        return
    topk_workspace = _selection_workspace(q.device)
    logits = qsa_mqa_paged_decode(
        q,
        k_cache,
        page_table,
        visible_blocks,
        decode_query_len,
    )
    _select_qsa_scores(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        topk_workspace,
    )


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score and select compressed prefill blocks in bounded chunks.

    Args:
        q: Packed prefill query tensor shaped ``[num_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        query_start_loc: Packed prefill query offsets with a terminal offset.
            Offsets may share the base of a larger backing tensor.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        max_query_len: Maximum number of query tokens in one request.
        block_indices: Compressed-index output buffer.
    """

    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    rows = q.shape[0]
    if not rows:
        return
    columns = page_table.shape[1] * k_cache.shape[1]
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    topk_workspace = _selection_workspace(q.device)

    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        query_slice = slice(query_start, query_end)
        logits = _launch_qsa_mqa_paged_prefill(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            max_query_len,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
        _select_qsa_scores(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
        )


__all__ = [
    "expand_qsa_block_indices",
    "qsa_mqa_paged_decode",
    "qsa_mqa_paged_prefill",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_paged_decode",
]
