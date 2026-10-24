# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Step-5 CSA kernels; each cache has its own block table and page size."""

import torch

from vllm.triton_utils import tl, triton


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
    k_scale_ptr,
    v_scale_ptr,
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
    FP8: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    req = tl.load(req_ptr + token).to(tl.int64)
    position = tl.load(pos_ptr + token)
    if FP8:
        scale = scale * tl.load(k_scale_ptr)

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
        if FP8:
            k = k.to(q.dtype)
            v = v.to(q.dtype)
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

    if FP8:
        acc = acc * tl.load(v_scale_ptr)
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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Attention over each query's selected regions of the paged K/V, into
    ``output``. ``key_cache``/``value_cache`` are ``[pages, page_tokens, kv_heads,
    head_dim]`` views; ``token_req`` and ``positions`` locate every query row."""
    tokens, heads, head_dim = query.shape
    if tokens == 0:
        return
    groups = key_cache.shape[2]
    heads_per_group = heads // groups
    fp8 = key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert value_cache.dtype == key_cache.dtype
    assert not fp8 or (k_scale is not None and v_scale is not None)
    _attention_paged_kernel[(tokens, groups)](
        query,
        key_cache,
        value_cache,
        regions,
        block_table,
        token_req,
        positions,
        output,
        k_scale,
        v_scale,
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
        FP8=fp8,
    )
