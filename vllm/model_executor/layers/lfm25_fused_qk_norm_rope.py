# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in fused Q/K RMSNorm + RoPE kernel for LFM2.5 attention."""

import logging
import os

import torch

from vllm.triton_utils import tl, triton

FUSED_QK_NORM_ROPE_ENABLED = os.getenv("VLLM_LFM25_FUSED_QK_NORM_ROPE", "0") == "1"
if FUSED_QK_NORM_ROPE_ENABLED:
    logging.getLogger(__name__).info("[LFM2.5] Q/K RMSNorm + RoPE fusion ENABLED")


@triton.jit
def _lfm25_fused_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    lh = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_p = k_ptr + pid * k_stride_t + lh * head_dim
        w_p = k_weight_ptr
        out_p = k_out_ptr + pid * k_out_stride_t + lh * head_dim
    else:
        in_p = q_ptr + pid * q_stride_t + lh * head_dim
        w_p = q_weight_ptr
        out_p = q_out_ptr + pid * q_out_stride_t + lh * head_dim

    hob = tl.arange(0, HEAD_BLOCK)
    hm = hob < head_dim
    x = tl.load(in_p + hob, mask=hm, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    ir = tl.rsqrt(var + eps)

    if HAS_PASS:
        w = tl.load(w_p + hob, mask=hm, other=0.0).to(tl.float32)
        norm = (x * ir * w).to(INPUT_DTYPE).to(tl.float32)
        pm = hm & (hob >= rotary_dim)
        tl.store(out_p + hob, norm, mask=pm)

    rob = tl.arange(0, ROT_HALF_BLOCK)
    rm = rob < half_rotary
    x_1 = tl.load(in_p + rob, mask=rm, other=0.0).to(tl.float32)
    x_2 = tl.load(in_p + half_rotary + rob, mask=rm, other=0.0).to(tl.float32)
    w_1 = tl.load(w_p + rob, mask=rm, other=0.0).to(tl.float32)
    w_2 = tl.load(w_p + half_rotary + rob, mask=rm, other=0.0).to(tl.float32)

    x_1 = (x_1 * ir * w_1).to(INPUT_DTYPE).to(tl.float32)
    x_2 = (x_2 * ir * w_2).to(INPUT_DTYPE).to(tl.float32)

    pos = tl.load(positions_ptr + pid).to(tl.int64)
    cb = pos * cache_stride_p
    cc = tl.load(cos_sin_cache_ptr + cb + rob, mask=rm, other=0.0).to(tl.float32)
    ss = tl.load(cos_sin_cache_ptr + cb + half_rotary + rob, mask=rm, other=0.0).to(
        tl.float32
    )

    tl.store(out_p + rob, x_1 * cc - x_2 * ss, mask=rm)
    tl.store(out_p + half_rotary + rob, x_2 * cc + x_1 * ss, mask=rm)


def fused_lfm25_qk_rmsnorm_rope(
    q,
    k,
    qw,
    kw,
    cs,
    pos,
    eps,
    nqh,
    nkvh,
    hd,
    rd,
):
    if not all(t.is_cuda and t.device == q.device for t in (q, k, qw, kw, cs, pos)):
        raise ValueError("all tensors must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[1] != nqh * hd or k.shape[1] != nkvh * hd:
        raise ValueError("token counts or head dimensions mismatch")
    if q.stride(1) != 1 or k.stride(1) != 1 or cs.stride(1) != 1:
        raise ValueError("tensors must have unit feature stride")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or cs.dtype != q.dtype
    ):
        raise ValueError("Q, K, cs must share BF16/FP16 dtype")
    if qw.numel() != hd or kw.numel() != hd or pos.shape[0] < q.shape[0]:
        raise ValueError("weights/position tensors shape mismatch")

    n = q.shape[0]
    qo = torch.empty((n, nqh * hd), dtype=q.dtype, device=q.device)
    ko = torch.empty((n, nkvh * hd), dtype=k.dtype, device=k.device)
    if n == 0:
        return qo, ko

    hb = triton.next_power_of_2(hd)
    rb = triton.next_power_of_2(rd // 2)
    _lfm25_fused_qk_norm_rope_kernel[(n, nqh + nkvh)](
        q,
        k,
        qo,
        ko,
        qw,
        kw,
        cs,
        pos,
        q.stride(0),
        k.stride(0),
        qo.stride(0),
        ko.stride(0),
        cs.stride(0),
        nqh,
        nkvh,
        hd,
        rd,
        rd // 2,
        eps,
        INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=hb,
        ROT_HALF_BLOCK=rb,
        HAS_PASS=rd < hd,
        num_warps=max(1, hb // 64),
        num_stages=2,
    )
    return qo, ko
