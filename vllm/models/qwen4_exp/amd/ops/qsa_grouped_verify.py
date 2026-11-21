# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped QSA indexer scorer for MTP verify rows on ROCm.

In a uniform verify batch each request has ``ROWS`` consecutive query rows
(1 + the number of speculative tokens), and they all read the same compressed
keys of that request. ``_qsa_mqa_paged_kernel`` scores every row on its own,
so it loads each key tile ``ROWS`` times and reduces the dot products with
``tl.sum`` on vector units. This kernel gives one program a group of ``ROWS``
rows and a run of column tiles: it gathers a key tile once and scores all
``ROWS x NUM_HEADS`` queries against it with one ``tl.dot``:

    score[r, c] = sum_h relu(q[r, h, :] . k[c, :]) / score_divisor

It writes what ``_qsa_mqa_paged_kernel`` writes: ``visible`` per row, the
same validity rules, and the columns past a row's ``visible`` count left
unwritten. Only the order of the fp32 sums differs, as in the MFMA prefill
scorer (``qsa_mfma_scorer.py``).

A group whose rows do not all belong to one request (a short or padded
batch) still gives correct scores: keys are gathered for the group's first
valid request, and a row of another request is masked out of the store and
scored by a second pass over its own request. The caller uses this kernel
only for decode rows, where every request has the same query length.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Score programs a launch aims for. Each program strides over the column tiles
# of its group, so the grid does not depend on the context length and a
# captured graph replays at any length.
_GROUPED_PROGRAMS = 2048
_BLOCK_N = 64
_NUM_WARPS = 4


@triton.jit(
    do_not_specialize=["num_rows", "num_requests", "num_columns"],
    do_not_specialize_on_alignment=[
        "token_to_req_ptr",
        "query_positions_ptr",
        "visible_blocks_ptr",
    ],
)
def _qsa_mqa_paged_grouped_kernel(
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
    ROWS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    first_tile = tl.program_id(1)
    # Query index qi = row_in_group * NUM_HEADS + head, padded to BLOCK_Q.
    qi = tl.arange(0, BLOCK_Q)
    q_row_in_group = qi // NUM_HEADS
    q_head = qi % NUM_HEADS
    q_rows = group * ROWS + q_row_in_group
    q_live = (q_row_in_group < ROWS) & (q_rows < num_rows)
    dims = tl.arange(0, BLOCK_D)

    # Per-row metadata, also in the [BLOCK_Q] layout (one entry per query).
    request = tl.load(token_to_req_ptr + q_rows, mask=q_live, other=-1)
    position = tl.load(query_positions_ptr + q_rows, mask=q_live, other=-1)
    request_ok = q_live & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request, mask=request_ok, other=0
    )
    visible = tl.minimum(
        (position + 1) // COMPRESS_RATIO, sequence_length // COMPRESS_RATIO
    )
    if first_tile == 0:
        tl.store(visible_blocks_ptr + q_rows, visible, mask=q_live & (q_head == 0))
    end = tl.where(request_ok, tl.minimum(visible, num_columns), 0)

    query = tl.load(
        q_ptr
        + q_rows[:, None] * stride_q_row
        + q_head[:, None] * stride_q_head
        + dims[None, :] * stride_q_dim,
        mask=q_live[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    # [BLOCK_Q, BLOCK_Q] 0/1 matrix that sums the heads of a row: the score
    # of row r is sum over qi with q_row_in_group[qi] == r.
    head_sum = (q_row_in_group[:, None] == q_row_in_group[None, :]).to(tl.float32)

    # Rows of the group are scored per distinct request; in a uniform verify
    # batch this loop runs once.
    remaining = request_ok
    for _ in range(ROWS):
        if tl.max(remaining.to(tl.int32), axis=0) > 0:
            current = tl.max(tl.where(remaining, request, -1), axis=0)
            in_run = remaining & (request == current)
            remaining = remaining & (request != current)
            run_end = tl.max(tl.where(in_run, end, 0), axis=0)
            q_run = tl.where(in_run[:, None], query, 0.0)
            for tile in range(
                first_tile, tl.cdiv(run_end, BLOCK_N), tl.num_programs(1)
            ):
                columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
                logical_page = columns // PAGE_SIZE
                page_offset = columns % PAGE_SIZE
                live = (columns < run_end) & (logical_page < PAGE_TABLE_WIDTH)
                safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
                physical_page = tl.load(
                    page_table_ptr
                    + current * stride_table_req
                    + safe_logical_page * stride_table_page,
                    mask=live,
                    other=-1,
                )
                page_ok = (physical_page >= 0) & (physical_page < num_pages)
                # physical_page * block stride can overflow int32.
                safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
                keys = tl.load(
                    k_cache_ptr
                    + safe_physical_page[None, :] * stride_cache_block
                    + page_offset[None, :] * stride_cache_token
                    + dims[:, None] * stride_cache_dim,
                    mask=page_ok[None, :] & (dims[:, None] < HEAD_DIM),
                    other=0.0,
                )
                # [BLOCK_Q, BLOCK_N]: every (row, head) query against the tile.
                dot = tl.maximum(tl.dot(q_run, keys), 0.0)
                # Sum the heads of each row; every query entry of a row then
                # holds that row's score.
                score = tl.dot(head_sum, dot) / score_divisor
                store = (
                    in_run[:, None]
                    & (q_head[:, None] == 0)
                    & (columns[None, :] < end[:, None])
                )
                tl.store(
                    logits_ptr + q_rows[:, None] * stride_logits_row + columns[None, :],
                    tl.where(page_ok[None, :], score, -float("inf")),
                    mask=store,
                )


def grouped_verify_scorer_applies(
    q: torch.Tensor, k_cache: torch.Tensor, rows_per_request: int
) -> bool:
    """Whether a decode chunk of ``qsa_mqa_paged`` takes the grouped kernel."""
    return (
        2 <= rows_per_request <= 8
        and q.shape[0] % rows_per_request == 0
        and q.shape[1] * rows_per_request <= 32
        and q.dtype == k_cache.dtype
        and q.dtype in (torch.bfloat16, torch.float16)
    )


def launch_qsa_mqa_paged_grouped(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    score_divisor: float,
    rows_per_request: int,
) -> None:
    """Fill ``logits`` [rows, columns] and ``visible_blocks`` [rows] for decode
    rows that come in runs of ``rows_per_request`` rows per request."""
    logger.info_once(
        "QSA indexer: verify rows use the grouped scorer (%d rows a request)",
        rows_per_request,
    )
    rows, num_heads, head_dim = q.shape
    columns = logits.shape[1]
    groups = triton.cdiv(rows, rows_per_request)
    tiles = triton.cdiv(columns, _BLOCK_N)
    per_group = max(1, min(tiles, _GROUPED_PROGRAMS // groups))
    _qsa_mqa_paged_grouped_kernel[(groups, per_group)](
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
        score_divisor,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        ROWS=rows_per_request,
        BLOCK_Q=max(16, triton.next_power_of_2(num_heads * rows_per_request)),
        BLOCK_N=_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(head_dim),
        COMPRESS_RATIO=compress_ratio,
        num_warps=_NUM_WARPS,
    )
