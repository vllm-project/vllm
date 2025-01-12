# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fla.ops.utils import (
    autocast_custom_bwd, autocast_custom_fwd, chunk_global_cumsum,
    chunk_global_reversed_cumsum, contiguous)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV", "USE_GK", "USE_GV", "USE_G"],
)
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None
})
@triton.jit
def fused_recurrent_fwd_kernel(
    # B: B, H: H, T: T, D: d_head
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, K]
    v,  # value [B, H, T, V]
    g,  # log gate [B, H, T] or None
    gk,  # log gate [B, H, T, K] or None
    gv,  # log gate [B, H, T, V] or None
    o,  # output [NK, B, H, T, V]
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    s_qk_h,  # stride size: T * K
    s_vo_h,  # stride size: T * V
    scale,  # K ** -0.5
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,  # whether to reverse the recurrence
    USE_GK: tl.constexpr,  # whether to use gk
    USE_GV: tl.constexpr,  # whether to use gv
    USE_G: tl.constexpr,  # whether to use g
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr  # whether to store final state
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(
        tl.int64), tl.program_id(2).to(tl.int64)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)

    if USE_G:
        p_g = g + i_bh * T + ((T - 1) if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
            (T - 1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
            (T - 1) * V if REVERSE else 0)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[None, :])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[:, None])
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V
        if USE_G:
            p_g += -1 if REVERSE else 1

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV", "USE_GK", "USE_GV", "USE_G"],
)
@triton.heuristics({
    'USE_INITIAL_STATE':
    lambda args: args['h0'] is not None,
    'STORE_INITIAL_STATE_GRADIENT':
    lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT':
    lambda args: args['dht'] is not None
})
# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_bwd_kernel(
        # B: B, H: H, T: T, D: d_head
        # NV: number of split in the V dimension.
        # NK: number of split in the K dimension
        q,  # query [B, H, T, K]
        k,  # key [B, H, T, V]
        v,  # value [B, H, T, V]
        g,  # log gate [B, H, T]
        gk,  # log gate [B, H, T, K] \alpha
        gv,  # log gate [B, H, T, V] \bete
        do,  # gradient wrt output [B, H, T, V]
        dq,  # gradient wrt query [NV, B, H, T, K]
        dk,  # gradient wrt key [NV, B, H, T, K]
        dv,  # gradient wrt value [NK, B, H, T, V]
        dht,  # gradient wrt final hidden state [B, H, K, V]
        dh0,  # gradient wrt initial hidden state [B, H, K, V]
        h0,  # initial hidden state [B, H, K, V]
        s_qk_h,  # stride size: T * K
        s_vo_h,  # stride size: T * V
        scale,  # K ** -0.5
        B,
        H,
        T,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        REVERSE: tl.
    constexpr,  # whether to do autoregressive modeling in the reverse direction
        USE_GK: tl.constexpr,  # whether to use gk
        USE_GV: tl.constexpr,  # whether to use gv
        USE_G: tl.constexpr,  # whether to use g
        USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
        STORE_INITIAL_STATE_GRADIENT: tl.
    constexpr,  # whether to store gradient wrt initial state
        USE_FINAL_STATE_GRADIENT: tl.
    constexpr,  # whether to compute gradient wrt final state
):
    i_v, i_k, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(
        tl.int64), tl.program_id(2).to(tl.int64)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
            (T - 1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
            (T - 1) * V if REVERSE else 0)
    if USE_G:
        p_g = g + i_bh * T + ((T - 1) if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[None, :])
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        d_q = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += -K if REVERSE else K
        p_v += -V if REVERSE else V
        p_q += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_dq += -K if REVERSE else K
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V
        if USE_G:
            p_g += -1 if REVERSE else 1

    # sync threads
    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (
            (T - 1) * K if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (
            (T - 1) * V if not REVERSE else 0)
    if USE_G:
        p_g = g + i_bh * T + ((T - 1) if not REVERSE else 0)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_dht, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(T):
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(b_dh * b_v[None, :], axis=1)
        d_v = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gv)[None, :]
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dh *= tl.exp(b_g)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_q += K if REVERSE else -K
        p_k += K if REVERSE else -K
        p_v += V if REVERSE else -V
        p_do += V if REVERSE else -V
        p_dk += K if REVERSE else -K
        p_dv += V if REVERSE else -V
        if USE_GK:
            p_gk += K if REVERSE else -K
        if USE_GV:
            p_gv += V if REVERSE else -V
        if USE_G:
            p_g += 1 if REVERSE else -1

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_kv)


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                gk: torch.Tensor,
                gv: torch.Tensor,
                scale: Optional[float] = None,
                initial_state: Optional[torch.Tensor] = None,
                output_final_state: bool = False,
                reverse: bool = False):
        B, H, T, K = q.shape
        V = v.shape[-1]
        # default scale
        if scale is None:
            scale = K**-0.5

        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)

        h0 = initial_state
        if output_final_state:
            ht = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            ht = None

        grid = (NV, NK, B * H)
        fused_recurrent_fwd_kernel[grid](q,
                                         k,
                                         v,
                                         g,
                                         gk,
                                         gv,
                                         o,
                                         h0,
                                         ht,
                                         q.stride(1),
                                         v.stride(1),
                                         scale,
                                         B=B,
                                         H=H,
                                         T=T,
                                         K=K,
                                         V=V,
                                         BK=BK,
                                         BV=BV,
                                         USE_G=g is not None,
                                         USE_GK=gk is not None,
                                         USE_GV=gv is not None,
                                         REVERSE=reverse)

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, g, gk, gv, h0, o)
        ctx.scale = scale
        ctx.reverse = reverse
        return o.to(q.dtype), ht

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, gk, gv, h0, o = ctx.saved_tensors

        # not supported yet.

        if dht is not None:
            err = "Cannot load final state gradient and use gates at the same time"  # noqa E501
            if g is not None:
                assert g.requires_grad is False, err
            if gk is not None:
                assert gk.requires_grad is False, err
            if gv is not None:
                assert gv.requires_grad is False, err

        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = ctx.scale

        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        dq = q.new_empty(NV, B, H, T, K, dtype=torch.float32)
        dk = q.new_empty(NV, B, H, T, K, dtype=torch.float32)
        dv = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        dh0 = torch.empty_like(h0) if (h0 is not None) else None
        grid = (NV, NK, B * H)

        fused_recurrent_bwd_kernel[grid](q,
                                         k,
                                         v,
                                         g,
                                         gk,
                                         gv,
                                         do,
                                         dq,
                                         dk,
                                         dv,
                                         dht,
                                         dh0,
                                         h0,
                                         q.stride(1),
                                         v.stride(1),
                                         scale,
                                         B=B,
                                         H=H,
                                         T=T,
                                         K=K,
                                         V=V,
                                         BK=BK,
                                         BV=BV,
                                         USE_G=g is not None,
                                         USE_GK=gk is not None,
                                         USE_GV=gv is not None,
                                         REVERSE=ctx.reverse)
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        fn = (chunk_global_cumsum
              if ctx.reverse else chunk_global_reversed_cumsum)
        dgk = fn(dq * q.float() - dk * k.float()) if gk is not None else None
        dgv = fn(do.float() * o.float() -
                 dv * v.float()) if gv is not None else None
        dg = fn((dq * q.float() -
                 dk * k.float()).sum(-1)) if g is not None else None
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(
            v.dtype), dg, dgk, dgv, None, dh0, None, None


def fused_recurrent(q,
                    k,
                    v,
                    g=None,
                    gk=None,
                    gv=None,
                    scale=None,
                    initial_state=None,
                    output_final_state=False,
                    reverse=False):
    return FusedRecurrentFunction.apply(q, k, v, g, gk, gv, scale,
                                        initial_state, output_final_state,
                                        reverse)
