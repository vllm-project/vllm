# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fla.ops.utils import (
    autocast_custom_bwd, autocast_custom_fwd, chunk_global_reversed_cumsum,
    contiguous)


@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
        q,  # query [B, H, T, K]
        k,  # key [B, H, T, K]
        v,  # value [B, H, T, V]
        w,  # log gate [B, H, T, K]
        u,  # bonus [B, H, K]
        o,  # output [B, H, T, V]
        # initial hidden state initialization [B, H, K, V]
    h0,
        ht,  # final hidden state [B, H, K, V]
        s_k_h,  # stride size: T * K
        s_v_h,  # stride size: T * V
        scale,  # K ** -0.5
        B: tl.constexpr,
        H: tl.constexpr,
        T: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,  # BLOCK SIZE along the K dimension
        BV: tl.constexpr,  # BLOCK SIZE along the V dimension
        USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
        STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
        REVERSE: tl.
    constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dq(
        # B: B, H: H, T: T, D: d_head
        # NV: number of split in the V dimension.
        # NK: number of split in the K dimension
        k,  # key [B, H, T, V]
        v,  # value [B, H, T, V]
        w,  # log gate [B, H, T, K]
        u,  # bonus [B, H, K]
        do,  # gradient of output [B, H, T, V]
        dq,  # gradient of query [NV, B, H, T, K]
        dq_aux,  # gradient of query_aux [NV, B, H, T, K]

        # initial hidden state initialization [B, H, K, V]
    h0,
        s_k_h,  # stride size: T * K
        s_v_h,  # stride size: T * V
        scale,  # K ** -0.5
        B: tl.constexpr,  # B
        H: tl.constexpr,  # H
        T: tl.constexpr,  # T
        K: tl.constexpr,  # K
        V: tl.constexpr,  # V
        BK: tl.constexpr,  # BLOCK SIZE along the K dimension
        BV: tl.constexpr,  # BLOCK SIZE along the V dimension
        USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
        REVERSE: tl.
    constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_dq_aux = dq_aux + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(
        0, BK) + ((T - 1) * K if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK

    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_kv = b_k[None, :] * b_v[:, None]
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        h_q = b_h * b_do[:, None]
        b_dq = tl.sum(h_q + b_kv * b_u[None, :] * b_do[:, None], axis=0)
        b_dq *= scale
        b_dq_aux = tl.sum(h_q, axis=0)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_bk)
        tl.store(p_dq_aux,
                 b_dq_aux.to(p_dq_aux.dtype.element_ty),
                 mask=mask_bk)
        p_k += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K
        p_dq += -K if REVERSE else K
        p_dq_aux += -K if REVERSE else K


@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dkv(
        # B: B, H: H, T: T, D: d_head
        # NV: number of split in the V dimension.
        # NK: number of split in the K dimension
        q,  # query [B, H, T, K]
        k,  # key [B, H, T, V]
        v,  # value [B, H, T, V]
        w,  # log gate [B, H, T, K]
        u,  # bonus [B, H, K]
        do,  # gradient of output [B, H, T, V]
        dk,
        dk_aux,
        dv,
        dh0,

        # initial hidden state initialization [B, H, K, V]
        s_k_h,  # stride size: T * K
        s_v_h,  # stride size: T * V
        scale,  # K ** -0.5
        B: tl.constexpr,  # B
        H: tl.constexpr,  # H
        T: tl.constexpr,  # T
        K: tl.constexpr,  # K
        V: tl.constexpr,  # V
        BK: tl.constexpr,  # BLOCK SIZE along the K dimension
        BV: tl.constexpr,  # BLOCK SIZE along the V dimension
        USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
        REVERSE: tl.
    constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    p_dk_aux = dk_aux + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(
        0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * V if not REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * K if not REVERSE else 0)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)

    for _ in range(T - 1, -1, -1):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_dkv = b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        tl.store(p_dk_aux, b_dk.to(p_dk_aux.dtype.element_ty), mask=mask_bk)
        b_dk += tl.sum(b_dkv * b_u[:, None] * b_v[None, :], axis=1)
        b_dv = tl.sum((b_dh + (b_dkv * b_u[:, None])) * b_k[:, None], axis=0)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_bv)
        b_dh *= tl.exp(b_w)[:, None]
        b_dh += b_dkv

        p_q += K if REVERSE else -K
        p_k += K if REVERSE else -K
        p_v += V if REVERSE else -V
        p_w += K if REVERSE else -K
        p_do += V if REVERSE else -V
        p_dk += K if REVERSE else -K
        p_dk_aux += K if REVERSE else -K
        p_dv += V if REVERSE else -V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(
            0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_kv)


class FusedRecurrentRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx,
                r: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                w: torch.Tensor,
                u: torch.Tensor,
                scale: Optional[float] = None,
                initial_state: Optional[torch.Tensor] = None,
                output_final_state: bool = False,
                reverse: bool = False):
        q = r
        B, H, T, K = q.shape
        V = v.shape[-1]

        BK, BV = min(triton.next_power_of_2(K),
                     32), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1

        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        grid = (NV, NK, B * H)
        fused_recurrent_rwkv6_fwd_kernel[grid](q,
                                               k,
                                               v,
                                               w,
                                               u,
                                               o,
                                               initial_state,
                                               final_state,
                                               k.stride(1),
                                               v.stride(1),
                                               scale,
                                               B=B,
                                               H=H,
                                               T=T,
                                               K=K,
                                               V=V,
                                               BK=BK,
                                               BV=BV,
                                               USE_INITIAL_STATE=initial_state
                                               is not None,
                                               STORE_FINAL_STATE=final_state
                                               is not None,
                                               REVERSE=reverse,
                                               num_warps=num_warps,
                                               num_stages=num_stages)

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, w, u, initial_state = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = ctx.scale

        BK, BV = min(triton.next_power_of_2(K),
                     16), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1
        dq = q.new_empty(NV, B, H, T, K, dtype=torch.float32)
        dq_aux = torch.empty_like(dq)
        grid = (NV, NK, B * H)

        fused_recurrent_rwkv6_bwd_kernel_dq[grid](
            k,
            v,
            w,
            u,
            do,
            dq,
            dq_aux,
            initial_state,
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
            USE_INITIAL_STATE=initial_state is not None,
            REVERSE=ctx.reverse,
            num_warps=num_warps,
            num_stages=num_stages)
        dq = dq.sum(0).to(q)
        dq_aux = dq_aux.sum(0)

        BK, BV = min(triton.next_power_of_2(K),
                     32), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        dk = q.new_empty(NV, B, H, T, K, dtype=torch.float32)
        dk_aux = q.new_empty(NV, B, H, T, K, dtype=torch.float32)
        dv = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        dh0 = initial_state.new_empty(B, H, K,
                                      V) if initial_state is not None else None
        grid = (NV, NK, B * H)
        fused_recurrent_rwkv6_bwd_kernel_dkv[grid](
            q,
            k,
            v,
            w,
            u,
            do,
            dk,
            dk_aux,
            dv,
            dh0,
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
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
            REVERSE=ctx.reverse,
        )
        dk = dk.sum(0).to(k)
        dv = dv.sum(0).to(v)
        dk_aux = dk_aux.sum(0)

        dw = (dq_aux * q * scale)[:, :, 1:] - (dk_aux * k)[:, :, 0:-1]
        dw = torch.nn.functional.pad(dw, (0, 0, 0, 1, 0, 0, 0, 0), value=0)
        dw = chunk_global_reversed_cumsum(dw).to(w)

        du = ((do * v).sum(-1)[..., None] * k * q * scale).sum([0, -2]).to(u)
        return dq, dk, dv, dw, du, None, dh0, None, None


def fused_recurrent_rwkv6(
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        scale: float = -1,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `(B, H, T, K)`. 
            Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            data-dependent decays of shape `(B, H, T, K)` 
            in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `(H, K)`
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape 
            `(B, H, K, V)`. Default: `False`.
    """
    if scale == -1:
        scale = r.shape[-1]**-0.5
    o, final_state = FusedRecurrentRWKV6Function.apply(r, k, v, w, u, scale,
                                                       initial_state,
                                                       output_final_state)
    return o, final_state
