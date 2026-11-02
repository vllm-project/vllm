# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper query-tiled sparse prefill attention for MiniMax M3."""

import torch

from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _FP8_DTYPES,
    _KV_SCALE_NONE,
    SPARSE_BLOCK_SIZE,
    _kv_scale_args,
)
from vllm.triton_utils import tl, triton

# Tuned on H200. Hopper selects this backend automatically.
_PREFILL_TILE_Q = 16


# ---------------------------------------------------------------------------
# Query-tiled GQA block-sparse attention (paged). Main heads attend only to the
# selected blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tiled_kernel(
    q_ptr,
    kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    sm_scale,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kv_blk: tl.constexpr,
    stride_kv_h: tl.constexpr,
    stride_kv_pos: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_ks_h: tl.constexpr,
    stride_ks_t: tl.constexpr,
    stride_vs_h: tl.constexpr,
    stride_vs_t: tl.constexpr,
    stride_th: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_bt_b: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
):
    BLOCK_SIZE_D: tl.constexpr = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H: tl.constexpr = triton.next_power_of_2(gqa_group_size)
    BLOCK_SIZE_QH: tl.constexpr = BLOCK_SIZE_Q * BLOCK_SIZE_H
    BLOCK_SIZE_T: tl.constexpr = triton.next_power_of_2(max_topk)

    pid_q = tl.program_id(0) * BLOCK_SIZE_Q
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if pid_q >= q_len:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    off_q = pid_q + tl.arange(0, BLOCK_SIZE_Q)
    q_abs = prefix_len + off_q
    off_t = tl.arange(0, BLOCK_SIZE_T)
    real_topk = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    END: tl.constexpr = 0x7FFFFFFF
    ids = tl.load(
        t_ptr
        + pid_kh * stride_th
        + (q_start + off_q[:, None]) * stride_tn
        + off_t[None, :] * stride_tk,
        mask=(off_q[:, None] < q_len) & (off_t[None, :] < real_topk[:, None]),
        other=END,
    ).to(tl.int32)
    ids = tl.where(ids >= 0, ids, END)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_SIZE_QH, BLOCK_SIZE_D))
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + pid_b * stride_bt_b
    sm_scale_log2e = sm_scale * 1.4426950409
    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), tl.float32)
    lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), tl.float32)
    blk = tl.min(tl.reshape(ids, (BLOCK_SIZE_Q * BLOCK_SIZE_T,)), axis=0)
    while blk < END:
        member = tl.sum((ids == blk).to(tl.int32), axis=1) > 0
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_n
        pos_mask = pos < seq_len
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                k_scale = tl.load(
                    k_scale_ptr
                    + pid_kh * stride_ks_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                    mask=pos_mask,
                    other=1.0,
                )
                k = (k * k_scale[None, :]).to(q.dtype)
        mask = (
            member[:, None, None]
            & (q_abs[:, None, None] >= pos[None, None, :])
            & pos_mask[None, None, :]
        )
        mask = tl.reshape(
            tl.broadcast_to(mask, (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K)),
            (BLOCK_SIZE_QH, BLOCK_SIZE_K),
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # A row may not select any of the union blocks visited so far. Avoid
        # -inf - -inf without giving these masked rows any softmax mass.
        safe_m = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.exp2(qk - safe_m[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - safe_m)[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                v_scale = tl.load(
                    v_scale_ptr
                    + pid_kh * stride_vs_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask,
                    other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = safe_m + tl.log2(tl.exp2(lse_i - safe_m) + l_ij)
        remaining = tl.where(ids > blk, ids, END)
        blk = tl.min(tl.reshape(remaining, (BLOCK_SIZE_Q * BLOCK_SIZE_T,)), axis=0)
    norm = tl.where(lse_i == float("-inf"), 0.0, tl.exp2(m_i - lse_i))
    acc_o *= norm[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(
        o_ptrs,
        tl.reshape(acc_o, (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)).to(
            o_ptr.dtype.element_ty
        ),
        boundary_check=(0, 1, 2),
    )


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Hopper query-tiled GQA sparse prefill over the selected blocks."""
    _, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    tile_q = _PREFILL_TILE_Q
    num_warps = min(16, max(4, tile_q * triton.next_power_of_2(gqa_group_size) // 16))
    grid = (triton.cdiv(max_query_len, tile_q), num_kv_heads, batch)
    _gqa_sparse_fwd_tiled_kernel[grid](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        gqa_group_size,
        head_dim,
        topk,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=tile_q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        num_warps=num_warps,
    )
