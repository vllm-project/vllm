# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm paged attention with Inkling's query-dependent relative bias.

The NVIDIA implementation uses the score-mod hook in tml-fa4.  ROCm Flash
Attention and AITER do not expose an equivalent hook, so this module implements
the same operation directly in Triton.  Query heads belonging to one KV head
are processed together and KV pages are gathered through vLLM's block table.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Keep the NVIDIA-facing split heuristic as API-compatible metadata.

    The ROCm Triton implementation performs online softmax in one program and
    does not consume the result.  Keeping this function unchanged avoids
    platform-specific scheduling branches in :mod:`inkling.amd.attention`.
    """
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    target_ctas = (
        256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    )
    max_splits = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            max_splits = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            max_splits = 32
        elif max_kv_len <= 65536:
            max_splits = 64
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: args["BLOCK_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["cache_seqlens", "cu_seqlens_q"])
def _inkling_rel_attention_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    out_ptr,  # [total_q, Hq, D]
    block_table_ptr,  # [batch, max_pages]
    cache_seqlens,
    cu_seqlens_q,
    gqa_group_size,
    head_dim,
    softmax_scale,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_p,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_p,
    stride_v_h,
    stride_v_d,
    stride_r_t,
    stride_r_h,
    stride_r_e,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_bt_b,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
):
    # A program owns BLOCK_Q query positions and every query head sharing one
    # KV head. This makes the QK/PV operations MFMA-friendly on CDNA.
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block = pid_q * BLOCK_Q
    if q_block >= q_len:
        return

    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_q_t, stride_q_h, stride_q_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))

    q_rows = q_block + tl.arange(0, BLOCK_Q)
    q_abs = prefix_len + q_rows
    q_valid = q_rows < q_len
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for k_start in range(0, kv_len, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_pos < kv_len
        # Masked lanes still need an in-range page-table load.
        safe_pos = tl.minimum(k_pos, tl.maximum(kv_len - 1, 0))
        page = tl.load(bt_row + safe_pos // page_size).to(tl.int64)
        page_off = safe_pos % page_size

        k = tl.load(
            k_ptr
            + page[None, :] * stride_k_b
            + page_off[None, :] * stride_k_p
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * (softmax_scale * log2e)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left

        # rel_logits is query- and head-dependent. Expand [Q, K] distance
        # indices across the GQA heads, then flatten to match QK's row order.
        off_h = tl.arange(0, BLOCK_H)
        rel_dist = dist[:, None, :]
        rel_valid = (
            q_valid[:, None, None]
            & (off_h[None, :, None] < gqa_group_size)
            & (rel_dist >= 0)
            & (rel_dist < rel_extent)
        )
        safe_dist = tl.maximum(0, tl.minimum(rel_dist, rel_extent - 1))
        bias = tl.load(
            rel_ptr
            + (q_start + q_rows[:, None, None]) * stride_r_t
            + (q_head_start + off_h[None, :, None]) * stride_r_h
            + safe_dist * stride_r_e,
            mask=rel_valid,
            other=0.0,
        )
        bias = tl.reshape(bias, (BLOCK_QH, BLOCK_K))
        scores += bias.to(tl.float32) * log2e
        score_valid = tl.reshape(
            score_valid[:, None, :] & (off_h[None, :, None] < gqa_group_size),
            (BLOCK_QH, BLOCK_K),
        )
        scores = tl.where(score_valid, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        has_scores = tl.sum(score_valid.to(tl.int32), axis=1) > 0
        # A sliding-window query can have entire early KV tiles masked. Avoid
        # the -inf - -inf NaN in online softmax until its first live tile.
        alpha = tl.where(has_scores, tl.exp2(m_i - m_ij), 1.0)
        p = tl.where(
            score_valid,
            tl.exp2(scores - m_ij[:, None]),
            0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]

        v = tl.load(
            v_ptr
            + page[:, None] * stride_v_b
            + page_off[:, None] * stride_v_p
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=k_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc /= l_i[:, None]
    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + q_start * stride_o_t + q_head_start * stride_o_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_o_t, stride_o_h, stride_o_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    tl.store(
        out_block_ptr,
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


@torch.no_grad()
def inkling_fa4_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged varlen attention with Inkling's relative score modification."""
    del num_splits
    if not causal or window_size[1] not in (0, -1):
        raise NotImplementedError("Inkling ROCm attention requires causal masking")
    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected q [T,H,D] and paged K/V [B,P,Hkv,D]")
    if rel_logits.shape != (q.shape[0], q.shape[1], rel_extent):
        raise ValueError(
            f"relative logits have shape {tuple(rel_logits.shape)}, expected "
            f"{(q.shape[0], q.shape[1], rel_extent)}"
        )

    num_kv_heads = key_cache.shape[2]
    if q.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if out is None:
        out = torch.empty_like(q)
    gqa_group_size = q.shape[1] // num_kv_heads
    batch = cu_seqlens_q.shape[0] - 1
    block_q = 1 if max_seqlen_q == 1 else 4
    grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, batch)
    _inkling_rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        out,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        gqa_group_size,
        q.shape[2],
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        page_size=key_cache.shape[1],
        rel_extent=rel_extent,
        window_left=window_size[0],
        BLOCK_Q=block_q,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
    )
    return out
