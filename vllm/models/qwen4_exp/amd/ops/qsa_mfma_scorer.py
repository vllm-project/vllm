# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MFMA tile scorer for the Qwen4Exp QSA indexer's prefill chunks.

``_qsa_mqa_paged_kernel`` (``qsa.py``) gives each query row its own programs
and reduces every (row, column, head) dot product with ``tl.sum``. Prefill rows
of one request read the same compressed keys, so this kernel scores a
``BLOCK_M x BLOCK_N`` tile per program with ``tl.dot`` (bf16/fp16 operands,
fp32 accumulation) and gathers each key tile once for all the tile's rows:

    score[r, c] = sum_h relu(q[r, h, :] . k[c, :]) / score_divisor

It computes what ``_qsa_mqa_paged_kernel`` computes: ``visible`` per row, the
same validity rules, the heads summed in order, and the columns past a row's
``visible`` count left unwritten. Only the summation order of each dot product
differs, so the scores are not bitwise equal: both kernels form the bf16 or
fp16 products exactly in fp32 and differ only in the order of the fp32 sums.

A tile whose rows belong to several requests scores them one request after
another, so a tile of decode rows (one request each) costs up to ``BLOCK_M``
passes; ``qsa_mqa_paged`` uses this kernel only for calls of at least
``MFMA_SCORER_MIN_ROWS`` rows that its caller marks as prefill.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Below 768 rows the launch costs more than the MFMA saves: on MI355X, against
# the per-row kernel's earlier one-program-per-32-columns grid, the selection
# (scores, top-k, expansion) of a 512-row chunk tied and 768 rows was the first
# clear win.
MFMA_SCORER_MIN_ROWS = 768
_BLOCK_M = 64
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 1


@triton.jit(
    do_not_specialize=["num_rows", "num_requests"],
    # The selection passes its chunks at a row offset (after the decode
    # rows), so the alignment of these pointers changes from step to step.
    # The kernel gathers from them element by element, and an alignment key
    # would compile another variant whenever the offset is not a multiple of
    # 16 bytes.
    do_not_specialize_on_alignment=[
        "page_table_ptr",
        "token_to_req_ptr",
        "query_positions_ptr",
        "sequence_lengths_ptr",
    ],
)
def _qsa_mqa_paged_mfma_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    tile_rows = tl.arange(0, BLOCK_M)
    rows = tl.program_id(0) * BLOCK_M + tile_rows
    row_in = rows < num_rows
    first_column = tl.program_id(1) * BLOCK_N
    columns = first_column + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)

    request = tl.load(token_to_req_ptr + rows, mask=row_in, other=-1)
    query_position = tl.load(query_positions_ptr + rows, mask=row_in, other=-1)
    request_ok = row_in & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request, mask=request_ok, other=0
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=row_in)
    # The top-k reads a row's first ``visible`` columns only; the columns past
    # them are left unwritten, and a column tile past every row's end does
    # nothing.
    end = tl.where(request_ok, tl.minimum(visible, num_columns), 0)
    if first_column >= tl.max(end, axis=0):
        return

    # Rows that follow a row of another request start a new run; the loop
    # visits each distinct request of the tile once, and a tile of N runs
    # holds at most N distinct requests.
    previous_request = tl.load(
        token_to_req_ptr + rows - 1, mask=row_in & (tile_rows > 0), other=-1
    )
    num_runs = 1 + tl.sum(
        ((tile_rows > 0) & row_in & (request != previous_request)).to(tl.int32),
        axis=0,
    )
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    valid = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int1)
    remaining = row_in
    for _ in range(num_runs):
        current = tl.max(tl.where(remaining, request, -2147483647), axis=0)
        in_group = remaining & (request == current)
        remaining = remaining & (request != current)
        safe_current = tl.minimum(tl.maximum(current, 0), num_requests - 1)
        # 0 for a run of rows without a valid request.
        group_end = tl.max(tl.where(in_group, end, 0), axis=0)
        live = (columns < group_end) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            page_table_ptr
            + safe_current * stride_table_req
            + safe_logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_ok = (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_ok[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        keys_t = tl.trans(keys)
        group_score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for head in tl.static_range(0, NUM_HEADS):
            query = tl.load(
                q_ptr
                + rows[:, None] * stride_q_row
                + head * stride_q_head
                + dims[None, :] * stride_q_dim,
                mask=in_group[:, None] & (dims[None, :] < HEAD_DIM),
                other=0.0,
            )
            group_score += tl.maximum(tl.dot(query, keys_t), 0.0)
        score = tl.where(in_group[:, None], group_score, score)
        valid = valid | (in_group[:, None] & page_ok[None, :])

    score /= score_divisor
    tl.store(
        logits_ptr + rows[:, None] * stride_logits_row + columns[None, :],
        tl.where(valid, score, -float("inf")),
        mask=columns[None, :] < end[:, None],
    )


def mfma_scorer_applies(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Whether a prefill call of ``qsa_mqa_paged`` takes the MFMA kernel."""
    return (
        q.shape[0] >= MFMA_SCORER_MIN_ROWS
        and q.dtype == k_cache.dtype
        and q.dtype in (torch.bfloat16, torch.float16)
    )


def launch_qsa_mqa_paged_mfma(
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
) -> None:
    """Fill ``logits`` [rows, columns] and ``visible_blocks`` [rows] from the
    arguments ``qsa_mqa_paged`` has checked and allocated."""
    logger.info_once(
        "QSA indexer: prefill chunks of >= %d rows use the MFMA scorer",
        MFMA_SCORER_MIN_ROWS,
    )
    columns = logits.shape[1]
    _qsa_mqa_paged_mfma_kernel[
        (triton.cdiv(q.shape[0], _BLOCK_M), triton.cdiv(columns, _BLOCK_N))
    ](
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
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        score_divisor,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        COMPRESS_RATIO=compress_ratio,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )


def warmup_qsa_mqa_paged_mfma(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    compress_ratio: int,
    max_num_batched_tokens: int,
) -> tuple[int, ...]:
    """Compile, without launching, the variants serving can reach and return
    the score widths compiled.

    ``qsa_select_paged_tokens`` scores at a multiple of 64 columns, or at the
    page table's capacity under graph capture and for a context near the
    maximum. The width (also the logits row stride) is the only argument whose
    specialization changes between steps; ``page_table`` and ``k_cache`` are
    the serving tensors, so their strides and storage size match.
    """
    capacity = page_table.shape[1] * k_cache.shape[1]
    if max_num_batched_tokens < MFMA_SCORER_MIN_ROWS or k_cache.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return ()
    rows = MFMA_SCORER_MIN_ROWS
    q = TritonWarmupTensor(k_cache.dtype, shape=(rows, num_heads, head_dim))
    index = TritonWarmupTensor(torch.int32, shape=(rows,))
    positions = TritonWarmupTensor(torch.int64, shape=(rows,))
    sequence_lengths = TritonWarmupTensor(torch.int32, shape=(page_table.shape[0],))
    widths = tuple(sorted({min(64, capacity), capacity}))
    for columns in widths:
        _qsa_mqa_paged_mfma_kernel.warmup(
            q,
            k_cache,
            page_table,
            index,
            positions,
            sequence_lengths,
            index,
            TritonWarmupTensor(torch.float32, shape=(rows, columns)),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(3),
            page_table.stride(0),
            page_table.stride(1),
            columns,
            rows,
            columns,
            k_cache.shape[0],
            page_table.shape[0],
            1.0,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=triton.next_power_of_2(head_dim),
            COMPRESS_RATIO=compress_ratio,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
            grid=(1, 1),
        )
    return widths


__all__ = [
    "MFMA_SCORER_MIN_ROWS",
    "launch_qsa_mqa_paged_mfma",
    "mfma_scorer_applies",
    "warmup_qsa_mqa_paged_mfma",
]
