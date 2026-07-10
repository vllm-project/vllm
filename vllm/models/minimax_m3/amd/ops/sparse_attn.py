# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm gfx942/gfx950 block-sparse GQA prefill kernel for MiniMax-M3.

Only the prefill path is specialized on CDNA: each 128-token KV block is split
into SUB_K-token sub-tiles to right-size the per-block QK/PV MFMAs. Everything
else -- the decode split-K kernels, the FP8 dtype set, the sparse block size --
is reused unchanged from ``common.ops.sparse_attn``.
"""

import torch

from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _FP8_DTYPES,
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms.rocm import on_gfx950, on_mi3xx
from vllm.triton_utils import tl, triton

__all__ = ["minimax_m3_sparse_attn", "minimax_m3_sparse_attn_decode"]


# Sub-tile width for the prefill kernel's per-block QK/PV GEMMs. gfx950 -> 64,
# gfx942 -> 32 (re-tune with tune_sparse_attn.py). Must divide SPARSE_BLOCK_SIZE.
_SPARSE_ATTN_SUB_K = SPARSE_BLOCK_SIZE // 2 if on_gfx950() else SPARSE_BLOCK_SIZE // 4

_SPARSE_ATTN_PREFILL_KWARG: dict | None = None


def _sparse_attn_prefill_kwargs() -> dict:
    """MFMA + pipeline launch params for the sub-tiled prefill kernel.

    gfx942 and gfx950 share the same params: ``num_warps=1`` keeps one wave
    resident on the small per-sub-tile GEMM, ``matrix_instr_nonkdim=16`` /
    ``kpack=2`` select the MFMA_16x16 path, and ``num_stages=1`` fits LDS and is
    fastest in the sweep. Only the sub-tile width (``_SPARSE_ATTN_SUB_K``)
    differs by arch. Empty on other AMD archs. Cached: arch is fixed per process.
    """
    global _SPARSE_ATTN_PREFILL_KWARG
    if _SPARSE_ATTN_PREFILL_KWARG is None:
        kwarg: dict = {}
        if on_mi3xx():
            kwarg = {
                "num_warps": 1,
                "matrix_instr_nonkdim": 16,
                "kpack": 2,
                "num_stages": 1,
            }
        _SPARSE_ATTN_PREFILL_KWARG = kwarg
    return _SPARSE_ATTN_PREFILL_KWARG


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    SUB_K: tl.constexpr,  # CDNA only: KV sub-tile width (see _IS_MI3XX)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

        # CDNA: process each 128-token KV block in SUB_K-token sub-tiles so
        # each QK/PV MFMA is right-sized. Numerically equivalent to the dense
        # path below (flash-softmax reassociation).
        NUM_SUB: tl.constexpr = BLOCK_SIZE_K // SUB_K
        for _ in tl.range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            for sub_i in range(NUM_SUB):
                off_sub = tl.arange(0, SUB_K) + sub_i * SUB_K
                pos_sub = c + off_sub
                pos_mask_sub = pos_sub < seq_len
                k_sub = tl.load(
                    kv_base
                    + 0 * stride_kv_kv
                    + off_sub[None, :] * stride_kv_pos
                    + off_d[:, None] * stride_kv_d,
                    mask=d_mask[:, None] & pos_mask_sub[None, :],
                    other=0.0,
                )
                if USE_FP8:
                    k_sub = k_sub.to(q.dtype)
                off_q_sub = (
                    tl.arange(0, BLOCK_SIZE_Q)[:, None]
                    + pid_q_j * BLOCK_SIZE_Q
                    + prefix_len
                    - off_sub[None, :]
                )
                qk_sub = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, SUB_K), dtype=tl.float32)
                # causal: q_abs_pos - k_off >= block_start (c)
                qk_sub += tl.where(off_q_sub[:, None, :] >= c, 0, float("-inf"))
                qk_sub = tl.reshape(qk_sub, BLOCK_SIZE_QH, SUB_K)
                qk_sub += tl.dot(q, k_sub) * sm_scale_log2e
                qk_sub += tl.where(pos_mask_sub[None, :], 0, float("-inf"))
                m_ij = tl.maximum(m_i, tl.max(qk_sub, axis=1))
                p_sub = tl.exp2(qk_sub - m_ij[:, None])
                l_ij = tl.sum(p_sub, axis=1)
                acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
                v_sub = tl.load(
                    kv_base
                    + 1 * stride_kv_kv
                    + off_sub[:, None] * stride_kv_pos
                    + off_d[None, :] * stride_kv_d,
                    mask=pos_mask_sub[:, None] & d_mask[None, :],
                    other=0.0,
                )
                if USE_FP8:
                    v_sub = v_sub.to(q.dtype)
                acc_o += tl.dot(p_sub.to(v_sub.dtype), v_sub)
                m_i = m_ij
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        SUB_K=_SPARSE_ATTN_SUB_K,
        **_sparse_attn_prefill_kwargs(),
    )
