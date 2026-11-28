# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless Triton kernels (with PyTorch fallbacks) for HiSparse prefill.

Why dedicated prefill kernels instead of reusing decode's residency kernels:
- `hisparse_resolve_residency` + `hisparse_gather_compact` use a single-CTA
  shared-memory hash table (~228 KB SMEM) and `int16` slot indices (`<= 32768`),
  sized for decode / MTP (`<= 4` query tokens). Prefill batches span hundreds
  to thousands of query tokens and up to `1M+` prefix rows.
- Prefill staging already materializes a flat `[1, num_staged_rows]` destination
  layout (`HiSparsePrefillStagingPlan`). A 2D scatter bitmap kernel plus a
  64-row block-skipping resident gather suffices with zero shared-memory table.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv

__all__ = [
    "hisparse_build_selected_mask",
    "hisparse_gather_resident_rows",
]


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_topk",
        "max_num_blocks_per_req",
        "num_staged_rows",
        "bt_stride0",
        "ti_stride0",
    ]
)
def _hisparse_mark_selected_rows_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    selected_mask_ptr,
    num_reqs,
    num_topk,
    max_num_blocks_per_req,
    num_staged_rows,
    bt_stride0,
    ti_stride0,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    in_bounds = offs < num_topk

    req = tl.load(req_id_ptr + token_id)
    valid_req = (req >= 0) & (req < num_reqs)
    safe_req = tl.where(valid_req, req, 0)

    tok = tl.load(
        token_indices_ptr + token_id.to(tl.int64) * ti_stride0 + offs,
        mask=in_bounds,
        other=-1,
    )
    safe_tok = tl.maximum(tok, 0)
    block_id = safe_tok // BLOCK_SIZE
    inblock_off = safe_tok % BLOCK_SIZE
    valid_block = (
        in_bounds & valid_req & (tok >= 0) & (block_id < max_num_blocks_per_req)
    )
    staged_block = tl.load(
        block_table_ptr + safe_req.to(tl.int64) * bt_stride0 + block_id,
        mask=valid_block,
        other=-1,
    )
    staged_row = staged_block.to(tl.int64) * BLOCK_SIZE + inblock_off.to(tl.int64)
    is_valid = (
        valid_block
        & (staged_block >= 0)
        & (staged_row >= 0)
        & (staged_row < num_staged_rows)
    )
    tl.store(selected_mask_ptr + staged_row, 1, mask=is_valid)


def hisparse_build_selected_mask(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    logical_topk_indices: torch.Tensor,
    *,
    block_size: int,
    num_staged_rows: int,
) -> torch.Tensor:
    """Build a `[1, num_staged_rows]` int32 bitmap of staging rows hit by top-k."""
    assert req_ids.shape[0] == logical_topk_indices.shape[0]
    num_tokens, num_topk = logical_topk_indices.shape
    num_reqs, max_blocks = block_table.shape
    selected_mask = torch.zeros(
        (1, max(num_staged_rows, 0)),
        dtype=torch.int32,
        device=logical_topk_indices.device,
    )
    if (
        num_tokens == 0
        or num_topk == 0
        or num_staged_rows <= 0
        or num_reqs == 0
        or max_blocks == 0
    ):
        return selected_mask

    if not logical_topk_indices.is_cuda or not HAS_TRITON:
        req_long = req_ids.to(torch.long)
        tok_long = logical_topk_indices.to(torch.long)
        block_id = torch.div(tok_long, block_size, rounding_mode="floor")
        valid = (
            ((req_long >= 0) & (req_long < num_reqs))[:, None]
            & (tok_long >= 0)
            & (block_id < max_blocks)
        )
        staged_blk = block_table.to(torch.long)[
            req_long.clamp(0, num_reqs - 1)[:, None],
            block_id.clamp(0, max_blocks - 1),
        ]
        staged_row = staged_blk * block_size + (tok_long % block_size)
        valid &= (staged_blk >= 0) & (staged_row >= 0) & (staged_row < num_staged_rows)
        selected_mask[0].index_fill_(0, staged_row[valid].unique(), 1)
        return selected_mask

    req_ids_c = req_ids.contiguous()
    block_table_c = (
        block_table if block_table.stride(1) == 1 else block_table.contiguous()
    )
    topk_c = (
        logical_topk_indices
        if logical_topk_indices.stride(1) == 1
        else logical_topk_indices.contiguous()
    )
    block_n = 512
    grid = (num_tokens, cdiv(num_topk, block_n))
    _hisparse_mark_selected_rows_kernel[grid](
        req_ids_c,
        block_table_c,
        topk_c,
        selected_mask,
        num_reqs,
        num_topk,
        max_blocks,
        num_staged_rows,
        block_table_c.stride(0),
        topk_c.stride(0),
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return selected_mask


@triton.jit(
    do_not_specialize=[
        "num_dst_rows",
        "num_src_rows",
        "src_block_stride",
        "src_row_stride",
        "dst_row_stride",
        "row_words",
    ]
)
def _hisparse_gather_resident_rows_kernel(
    src_ptr,
    dst_ptr,
    row_ids_ptr,
    num_dst_rows,
    num_src_rows,
    src_block_stride,
    src_row_stride,
    dst_row_stride,
    row_words,
    SRC_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    base_dst = tl.program_id(0) * BLOCK_M
    offs_m = base_dst + tl.arange(0, BLOCK_M)
    chunk_rows = tl.load(row_ids_ptr + offs_m, mask=offs_m < num_dst_rows, other=-1)
    if tl.max(chunk_rows, axis=0) < 0:
        return

    offs_d = tl.arange(0, BLOCK_D)
    m = 0
    while m < BLOCK_M:
        dst_row = base_dst + m
        src_row = tl.load(row_ids_ptr + dst_row, mask=dst_row < num_dst_rows, other=-1)
        if (src_row >= 0) & (src_row < num_src_rows):
            src_block = src_row // SRC_BLOCK_SIZE
            src_off = src_row % SRC_BLOCK_SIZE
            src_base = (
                src_block.to(tl.int64) * src_block_stride
                + src_off.to(tl.int64) * src_row_stride
            )
            dst_base = dst_row.to(tl.int64) * dst_row_stride
            d_start = 0
            while d_start < row_words:
                cols = d_start + offs_d
                mask_d = cols < row_words
                vals = tl.load(src_ptr + src_base + cols, mask=mask_d)
                tl.store(dst_ptr + dst_base + cols, vals, mask=mask_d)
                d_start += BLOCK_D
        m += 1


def hisparse_gather_resident_rows(
    source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor
) -> None:
    """Copy valid (`row_ids >= 0`) GPU-resident rows into staging."""
    num_dst_rows = destination.shape[0]
    block_size = source.shape[1]
    num_src_rows = source.shape[0] * block_size
    if num_dst_rows == 0 or row_ids.numel() == 0 or num_src_rows == 0:
        return
    assert row_ids.shape == (num_dst_rows,)
    assert (
        source.shape[2] * source.element_size()
        == destination.shape[1] * destination.element_size()
    )

    if source.is_cuda and HAS_TRITON:
        row_bytes = destination.shape[1] * destination.element_size()
        if row_bytes == 0:
            return
        src_bytes = source.view(torch.uint8)
        dst_bytes = destination.view(torch.uint8)
        aligned_8b = (
            row_bytes % 8 == 0
            and src_bytes.data_ptr() % 8 == 0
            and dst_bytes.data_ptr() % 8 == 0
            and src_bytes.storage_offset() % 8 == 0
            and dst_bytes.storage_offset() % 8 == 0
            and src_bytes.stride(0) % 8 == 0
            and src_bytes.stride(1) % 8 == 0
            and dst_bytes.stride(0) % 8 == 0
        )
        src_words = src_bytes.view(torch.int64) if aligned_8b else src_bytes
        dst_words = dst_bytes.view(torch.int64) if aligned_8b else dst_bytes
        block_m = 64
        _hisparse_gather_resident_rows_kernel[(cdiv(num_dst_rows, block_m),)](
            src_words,
            dst_words,
            row_ids.to(torch.int32).contiguous(),
            num_dst_rows,
            num_src_rows,
            src_words.stride(0),
            src_words.stride(1),
            dst_words.stride(0),
            src_words.shape[2],
            SRC_BLOCK_SIZE=block_size,
            BLOCK_M=block_m,
            BLOCK_D=256,
            num_warps=4,
        )
        return

    valid = (row_ids >= 0) & (row_ids < num_src_rows)
    if not source.is_cuda:
        if not bool(valid.any()):
            return
        dst_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        rows = row_ids.index_select(0, dst_idx).to(torch.long)
        payload = source[rows // block_size, rows % block_size]
        if payload.dtype != destination.dtype:
            payload = payload.contiguous().view(destination.dtype)
        destination.index_copy_(0, dst_idx, payload)
        return
    rows = row_ids.to(torch.long).clamp(0, num_src_rows - 1)
    payload = source[rows // block_size, rows % block_size]
    if payload.dtype != destination.dtype:
        payload = payload.contiguous().view(destination.dtype)
    destination.copy_(torch.where(valid[:, None], payload, destination))
