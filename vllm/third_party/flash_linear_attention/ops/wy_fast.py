# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501

from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import WarmupChoices
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

from .index import prepare_chunk_indices
from .utils import FLA_CHUNK_SIZE, autotune_unless_cdna3, is_cdna3

_WU_STOCK = [
    triton.Config({}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in [2, 4, 8]
    for num_stages in [2, 3, 4]
]

_WU_CDNA3 = (
    triton.Config(
        {"matrix_instr_nonkdim": 16, "waves_per_eu": 2}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"matrix_instr_nonkdim": 16, "waves_per_eu": 2}, num_warps=2, num_stages=2
    ),
)


def _wu_cdna3_config(programs: int, device: int) -> triton.Config:
    small, large = _WU_CDNA3
    return small if programs < 2 * num_compute_units(device) else large


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@autotune_unless_cdna3(
    configs=_WU_STOCK,
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_beta = beta + bos * H + i_h + o_t * H
    p_g = g + (bos * H + i_h) + o_t * H
    p_A = A + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :]
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_g = tl.exp(tl.load(p_g, mask=m_t, other=0.0))

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_u = u + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = (
            k
            + (bos * Hg + i_h // (H // Hg)) * K
            + o_t[:, None] * (Hg * K)
            + o_k[None, :]
        )
        p_w = w + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


def _recompute_w_u_warmup_inputs(
    *, dtype: torch.dtype, H: int, Hg: int, K: int, V: int, BT: int = FLA_CHUNK_SIZE
) -> dict[str, Any]:
    # Straddles the threshold in _wu_cdna3_config so both configs are compiled.
    programs: Any = WarmupChoices(1, 2 * num_compute_units(0))
    return dict(
        k=TritonWarmupTensor(dtype, shape=(1, BT, Hg, K)),
        v=TritonWarmupTensor(dtype, shape=(1, BT, H, V)),
        beta=TritonWarmupTensor(dtype, shape=(1, BT, H)),
        w=TritonWarmupTensor(dtype, shape=(1, BT, H, K)),
        u=TritonWarmupTensor(dtype, shape=(1, BT, H, V)),
        A=TritonWarmupTensor(dtype, shape=(1, BT, H, BT)),
        g=TritonWarmupTensor(torch.float32, shape=(1, BT, H)),
        cu_seqlens=TritonWarmupTensor(torch.int32, shape=(2,)),
        chunk_indices=TritonWarmupTensor(torch.int32, shape=(1, 2)),
        T=BT,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_chunks=1,
        num_batch_heads=programs,
        device=0,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=recompute_w_u_fwd_kernel,
    warmup_inputs=_recompute_w_u_warmup_inputs,
)
def _recompute_w_u_launch(
    k: Any,
    v: Any,
    beta: Any,
    w: Any,
    u: Any,
    A: Any,
    g: Any,
    cu_seqlens: Any,
    chunk_indices: Any,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    BT: int,
    num_chunks: int,
    num_batch_heads: int,
    device: int,
) -> DispatchSpec:
    launch = {}
    if is_cdna3:
        programs = num_chunks * num_batch_heads
        launch = _wu_cdna3_config(programs, device).all_kwargs()
    return (
        (num_chunks, num_batch_heads),
        dict(T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=64, BV=64, **launch),
        (w, u),
    )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    return _recompute_w_u_launch(
        k=k,
        v=v,
        beta=beta,
        w=k.new_empty(B, T, H, K),
        u=torch.empty_like(v),
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_chunks=NT,
        num_batch_heads=B * H,
        device=k.device.index,
    )
