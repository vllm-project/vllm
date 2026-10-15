# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Step-5 CSA kernels; each cache has its own block table and page size."""

import torch

from vllm.models.step5.common.ops.csa import round_e4m3
from vllm.triton_utils import tl, triton


@triton.jit
def _compress_paged_kernel(
    index_ptr,
    summary_ptr,
    index_bt_ptr,
    summary_bt_ptr,
    work_ptr,
    stride_index_bt,
    stride_summary_bt,
    stride_index_page,
    stride_index_row,
    stride_summary_page,
    stride_summary_row,
    INDEX_ROWS: tl.constexpr,
    SUMMARY_ROWS: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
):
    w = tl.program_id(0)
    req = tl.load(work_ptr + 2 * w).to(tl.int64)
    if req < 0:
        return
    region = tl.load(work_ptr + 2 * w + 1)
    tokens = region * REGION_SIZE + tl.arange(0, REGION_SIZE)
    page = tl.load(index_bt_ptr + req * stride_index_bt + tokens // INDEX_ROWS)
    dims = tl.arange(0, PROXY_DIM)
    base = (
        index_ptr
        + page.to(tl.int64)[:, None] * stride_index_page
        + (tokens % INDEX_ROWS)[:, None] * stride_index_row
        + dims[None, :]
    )
    k = tl.load(base).to(tl.float32)
    z = tl.load(base + PROXY_DIM).to(tl.float32)
    weights = tl.exp(z - tl.max(z, axis=0)[None, :])
    summary = tl.sum(weights * k, axis=0) / tl.sum(weights, axis=0)
    # Same rounding as ``round_e4m3``.
    summary = summary.to(tl.bfloat16).to(tl.float8e4nv).to(tl.bfloat16)
    out_page = tl.load(
        summary_bt_ptr + req * stride_summary_bt + region // SUMMARY_ROWS
    )
    tl.store(
        summary_ptr
        + out_page.to(tl.int64) * stride_summary_page
        + (region % SUMMARY_ROWS) * stride_summary_row
        + dims,
        summary,
    )


def compress_regions_paged(
    index_cache: torch.Tensor,
    summary_cache: torch.Tensor,
    index_block_table: torch.Tensor,
    summary_block_table: torch.Tensor,
    work: torch.Tensor,
    *,
    region_size: int,
) -> None:
    """Summarise the ``(request, region)`` pairs in ``work`` into the summary cache.

    Caches are ``[pages, rows_per_page, width]`` views, each with its own block
    table; index rows are ``[k | z]``. ``work`` rows with request ``-1`` are
    skipped.
    """
    if work.shape[0] == 0:
        return
    _compress_paged_kernel[(work.shape[0],)](
        index_cache,
        summary_cache,
        index_block_table,
        summary_block_table,
        work,
        index_block_table.stride(0),
        summary_block_table.stride(0),
        index_cache.stride(0),
        index_cache.stride(1),
        summary_cache.stride(0),
        summary_cache.stride(1),
        INDEX_ROWS=index_cache.shape[1],
        SUMMARY_ROWS=summary_cache.shape[1],
        REGION_SIZE=region_size,
        PROXY_DIM=summary_cache.shape[-1],
    )


@triton.jit
def _scores_paged_kernel(
    q_ptr,
    w_ptr,
    summary_ptr,
    block_table_ptr,
    req_ptr,
    pos_ptr,
    out_ptr,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_w_token,
    stride_w_group,
    stride_summary_page,
    stride_summary_row,
    stride_bt,
    stride_out_token,
    stride_out_group,
    max_regions,
    HEADS_PER_GROUP: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    SUMMARY_ROWS: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    regions = tl.program_id(2) * BLOCK_R + tl.arange(0, BLOCK_R)
    req = tl.load(req_ptr + token).to(tl.int64)
    history = tl.load(pos_ptr + token) // REGION_SIZE
    live = regions < history
    dims = tl.arange(0, PROXY_DIM)
    page = tl.load(
        block_table_ptr + req * stride_bt + regions // SUMMARY_ROWS,
        mask=live,
        other=0,
    )
    s = tl.load(
        summary_ptr
        + page.to(tl.int64)[None, :] * stride_summary_page
        + (regions % SUMMARY_ROWS)[None, :] * stride_summary_row
        + dims[:, None],
        mask=live[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for head in tl.static_range(HEADS_PER_GROUP):
        q = tl.load(
            q_ptr
            + token.to(tl.int64) * stride_q_token
            + group * stride_q_group
            + head * stride_q_head
            + dims
        ).to(tl.float32)
        w = tl.load(
            w_ptr + token.to(tl.int64) * stride_w_token + group * stride_w_group + head
        )
        acc += tl.maximum(tl.sum(q[:, None] * s, axis=0), 0.0) * w
    tl.store(
        out_ptr
        + token.to(tl.int64) * stride_out_token
        + group * stride_out_group
        + regions,
        tl.where(live, acc, float("-inf")),
        mask=regions < max_regions,
    )


def indexer_scores_paged(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_req: torch.Tensor,
    positions: torch.Tensor,
    *,
    max_regions: int,
    region_size: int,
) -> torch.Tensor:
    """``[T, G, Hg, P]`` queries vs each row's request history -> ``[T, G,
    max_regions]`` scores; regions at or past a row's own region are ``-inf``."""
    tokens, groups, heads_per_group, proxy_dim = index_q.shape
    out = torch.full(
        (tokens, groups, max(max_regions, 1)),
        float("-inf"),
        device=index_q.device,
        dtype=torch.float32,
    )
    if tokens == 0 or max_regions == 0:
        return out
    q = round_e4m3(index_q).contiguous()
    w = weights.float().contiguous()
    block_r = 64
    _scores_paged_kernel[(tokens, groups, triton.cdiv(max_regions, block_r))](
        q,
        w,
        summary_cache,
        block_table,
        token_req,
        positions,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        w.stride(0),
        w.stride(1),
        summary_cache.stride(0),
        summary_cache.stride(1),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        max_regions,
        HEADS_PER_GROUP=heads_per_group,
        PROXY_DIM=proxy_dim,
        SUMMARY_ROWS=summary_cache.shape[1],
        REGION_SIZE=region_size,
        BLOCK_R=block_r,
    )
    return out


@triton.jit
def _attention_paged_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    regions_ptr,
    block_table_ptr,
    req_ptr,
    pos_ptr,
    out_ptr,
    scale,
    stride_q_token,
    stride_q_head,
    stride_k_page,
    stride_k_row,
    stride_k_head,
    stride_v_page,
    stride_v_row,
    stride_v_head,
    stride_r_token,
    stride_r_group,
    stride_bt,
    stride_o_token,
    stride_o_head,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    req = tl.load(req_ptr + token).to(tl.int64)
    position = tl.load(pos_ptr + token)

    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < HEADS_PER_GROUP
    dim_ok = dims < HEAD_DIM
    q = tl.load(
        q_ptr
        + token.to(tl.int64) * stride_q_token
        + (group * HEADS_PER_GROUP + heads)[:, None] * stride_q_head
        + dims[None, :],
        mask=head_ok[:, None] & dim_ok[None, :],
        other=0.0,
    )
    row_max = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    in_region = tl.arange(0, REGION_SIZE)

    for base in tl.range(0, SLOTS, BLOCK_S):
        slots = base + tl.arange(0, BLOCK_S)
        ids = tl.load(
            regions_ptr
            + token.to(tl.int64) * stride_r_token
            + group * stride_r_group
            + slots,
            mask=slots < SLOTS,
            other=-1,
        )
        cols = tl.reshape(
            ids[:, None] * REGION_SIZE + in_region[None, :], (BLOCK_S * REGION_SIZE,)
        )
        live = tl.reshape(
            (ids[:, None] >= 0) & (in_region[None, :] >= 0), (BLOCK_S * REGION_SIZE,)
        )
        live = live & (cols <= position)
        safe = tl.where(live, cols, 0)
        page = tl.load(
            block_table_ptr + req * stride_bt + safe // PAGE_TOKENS,
            mask=live,
            other=0,
        ).to(tl.int64)
        row = (safe % PAGE_TOKENS).to(tl.int64)
        kv_mask = live[:, None] & dim_ok[None, :]
        k = tl.load(
            k_cache_ptr
            + page[:, None] * stride_k_page
            + row[:, None] * stride_k_row
            + group * stride_k_head
            + dims[None, :],
            mask=kv_mask,
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr
            + page[:, None] * stride_v_page
            + row[:, None] * stride_v_row
            + group * stride_v_head
            + dims[None, :],
            mask=kv_mask,
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        s = tl.where(live[None, :], s, float("-inf"))
        tile_max = tl.maximum(row_max, tl.max(s, axis=1))
        empty = tile_max == float("-inf")
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp(row_max - shift))
        p = tl.exp(s - shift[:, None])
        row_sum = row_sum * rescale + tl.sum(p, axis=1)
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        row_max = tile_max

    out = tl.where(row_sum[:, None] > 0, acc / row_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + token.to(tl.int64) * stride_o_token
        + (group * HEADS_PER_GROUP + heads)[:, None] * stride_o_head
        + dims[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None] & dim_ok[None, :],
    )


def sparse_attention_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    regions: torch.Tensor,
    block_table: torch.Tensor,
    token_req: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    *,
    region_size: int,
    scale: float,
) -> None:
    """Attention over each query's selected regions of the paged K/V, into
    ``output``. ``key_cache``/``value_cache`` are ``[pages, page_tokens, kv_heads,
    head_dim]`` views; ``token_req`` and ``positions`` locate every query row."""
    tokens, heads, head_dim = query.shape
    if tokens == 0:
        return
    groups = key_cache.shape[2]
    heads_per_group = heads // groups
    _attention_paged_kernel[(tokens, groups)](
        query,
        key_cache,
        value_cache,
        regions,
        block_table,
        token_req,
        positions,
        output,
        scale,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        regions.stride(0),
        regions.stride(1),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        HEADS_PER_GROUP=heads_per_group,
        HEAD_DIM=head_dim,
        PAGE_TOKENS=key_cache.shape[1],
        REGION_SIZE=region_size,
        SLOTS=regions.shape[-1],
        BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        # 8 regions x 8 tokens x 256 padded dims fits gfx942's 64 KiB LDS.
        BLOCK_S=8,
        num_stages=1,
    )
