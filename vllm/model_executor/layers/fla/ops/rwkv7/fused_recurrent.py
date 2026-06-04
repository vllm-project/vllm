# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from ..op import exp
from ..utils import input_guard, use_cuda_graph


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    kk,
    a,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_r = r + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_w = w + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_a = a + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_kk = kk + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k

    p_o = o + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if IS_DECODE:
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
        b_act_a = -b_kk
        b_b = b_kk * b_a

        b_h = (
            exp(b_w)[:, None] * b_h
            + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
        )
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_r[:, None], 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    else:
        for i_t in range(0, T):
            b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
            b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
            b_act_a = -b_kk
            b_b = b_kk * b_a

            b_h = (
                exp(b_w)[:, None] * b_h
                + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
            )
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_r[:, None], 0)

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            p_r += (-1 if REVERSE else 1) * H * K
            p_w += (-1 if REVERSE else 1) * H * K
            p_k += (-1 if REVERSE else 1) * H * K
            p_v += (-1 if REVERSE else 1) * H * V
            p_a += (-1 if REVERSE else 1) * H * K
            p_kk += (-1 if REVERSE else 1) * H * K
            p_o += (-1 if REVERSE else 1) * H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@input_guard
def fused_recurrent_rwkv7_fwd(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float | None = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    IS_DECODE = T == 1

    h0 = initial_state
    ht = r.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    fused_recurrent_rwkv7_fwd_kernel[grid](
        r,
        w,
        k,
        v,
        kk,
        a,
        o,
        h0,
        ht,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        REVERSE=reverse,
        IS_DECODE=IS_DECODE,
    )
    return o, ht


def fused_mul_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute S_t = S_{t-1} (I + a_t b_t^T) + v_t k_t^T per token.

    Args:
        r:  shape ``[B, T, H, K]``
        w:  log-decay shape ``[B, T, H, K]``
        k:  shape ``[B, T, H, K]``
        v:  shape ``[B, T, H, V]``
        kk: shape ``[B, T, H, K]`` — pre-normalized (k * k_k) projection used
            both as ``a`` and ``b`` factors inside the kernel
            (a_t = -kk, b_t = kk * a)
        a:  shape ``[B, T, H, K]`` — scalar gate
        initial_state: ``[N, H, K, V]`` for ``N`` sequences (``B`` if
            ``cu_seqlens`` is None, else ``len(cu_seqlens) - 1``)
        output_final_state: if True returns ``ht`` of shape ``[N, H, K, V]``
        cu_seqlens: ``[N+1]`` cumulative sequence lengths (FlashAttention API).
            When provided, ``r.shape[0]`` must be 1 (packed batch).
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`. "
                "Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = r.shape[-1] ** -0.5
    o, final_state = fused_recurrent_rwkv7_fwd(
        r,
        w,
        k,
        v,
        kk,
        a,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    return o, final_state
