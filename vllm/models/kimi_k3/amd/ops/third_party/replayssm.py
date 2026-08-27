# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Ported from ATOM atom/model_ops/fla_ops/replayssm.py for vLLM Kimi-K3 KDA.

from __future__ import annotations

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

__all__ = [
    "PAD_SLOT_ID",
    "flush_threshold_ok",
    "replayssm_buffer_shapes",
    "replayssm_commit",
    "replayssm_sigmoid_gating_delta_rule",
]

PAD_SLOT_ID = -1


def flush_threshold_ok(cache_len: int, max_query_len: int) -> bool:
    return cache_len >= 2 * max_query_len


def replayssm_buffer_shapes(
    cache_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    is_kda: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        (cache_len, num_v_heads, head_k_dim),
        (cache_len, num_v_heads, head_v_dim),
        (cache_len, num_v_heads, head_k_dim) if is_kda else (cache_len, num_v_heads),
    )


@triton.jit(do_not_specialize=["N", "T_MAX", "CAP"])
def _replayssm_commit_kernel(
    write_pos,
    slot_idx,
    num_accepted,
    N,
    T_MAX,
    CAP,
):
    i_n = tl.program_id(0)
    if i_n >= N:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    h = tl.load(write_pos + slot)
    prev_flushed = h + 2 * T_MAX > CAP
    base = tl.where(prev_flushed, 0, h)
    tl.store(write_pos + slot, base + tl.load(num_accepted + i_n))


def replayssm_commit(
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    num_accepted: torch.Tensor,
    max_query_len: int,
    cache_len: int,
) -> None:
    n = slot_idx.numel()
    if n == 0:
        return
    _replayssm_commit_kernel[(n,)](
        write_pos,
        slot_idx,
        num_accepted,
        n,
        max_query_len,
        cache_len,
        num_warps=1,
    )


@triton.jit
def _kda_gate(A_log_ptr, a_ptr, dt_bias_ptr, mask_k, LOWER_BOUND, USE_LOWER_BOUND):
    x = tl.load(a_ptr, mask=mask_k, other=0.0).to(tl.float32) + tl.load(
        dt_bias_ptr, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_A = tl.load(A_log_ptr).to(tl.float32)
    if USE_LOWER_BOUND:
        return LOWER_BOUND * tl.sigmoid(tl.exp(b_A) * x)
    softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    return -tl.exp(b_A) * softplus_x


@triton.jit(do_not_specialize=["N", "T_TOT", "T_MAX", "CAP"])
def _replayssm_kda_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    LOWER_BOUND,
    N,
    T_TOT,
    T_MAX,
    CAP,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_ckpt_slot: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return

    h = tl.load(write_pos + slot).to(tl.int32)
    do_flush = h + 2 * T_MAX > CAP
    base = tl.where(do_flush, 0, h).to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_ckpt_hv = (
        ckpt + slot * stride_ckpt_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    )
    b_h = tl.load(p_ckpt_hv, mask=mask_h, other=0.0).to(tl.float32)

    for j in range(h):
        b_rg = tl.load(
            buf_g + slot * stride_bufg_slot + j * stride_bufg_pos + i_hv * K + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_h *= exp(b_rg)[None, :]
        b_rk = tl.load(
            buf_k + slot * stride_bufk_slot + j * stride_bufk_pos + i_hv * K + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_ru = tl.load(
            buf_u + slot * stride_bufu_slot + j * stride_bufu_pos + i_hv * V + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_h += b_ru[:, None] * b_rk[None, :]

    if do_flush:
        tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=mask_h)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_a = a + (bos * HV + i_hv) * K + o_k
    p_b = b + bos * HV + i_hv
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv * K + o_k

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = _kda_gate(p_A_log, p_a, p_dt_bias, mask_k, LOWER_BOUND, USE_LOWER_BOUND)
        b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h *= exp(b_g)[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        tl.store(
            p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mask_v
        )

        pos = base + i_t
        p_bu = buf_u + slot * stride_bufu_slot + pos * stride_bufu_pos + i_hv * V + o_v
        tl.store(p_bu, b_v.to(p_bu.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            p_bk = (
                buf_k + slot * stride_bufk_slot + pos * stride_bufk_pos + i_hv * K + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg = (
                buf_g + slot * stride_bufg_slot + pos * stride_bufg_pos + i_hv * K + o_k
            )
            tl.store(p_bg, b_g.to(p_bg.dtype.element_ty), mask=mask_k)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_a += stride_a_token
        p_b += stride_b_token


def replayssm_sigmoid_gating_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    o: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor:
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[1]
    if scale is None:
        scale = K**-0.5
    assert flush_threshold_ok(cap, max_query_len), (
        f"replayssm cache_len={cap} must be >= 2*max_query_len={2 * max_query_len}"
    )

    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "K must fit one block"
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)

    q, k, v, a, b = (x.contiguous() for x in (q, k, v, a, b))
    out = q.new_empty(1, T_tot, HV, V) if o is None else o.unsqueeze(0)

    _replayssm_kda_fwd_kernel[(NV, N * HV)](
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=scale,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        N=N,
        T_TOT=T_tot,
        T_MAX=max_query_len,
        CAP=cap,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0),
        stride_bufk_pos=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0),
        stride_bufu_pos=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0),
        stride_bufg_pos=buf_g.stride(1),
        stride_a_token=a.stride(-3),
        stride_b_token=b.stride(-2),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=1,
        num_stages=3,
    )
    return out
