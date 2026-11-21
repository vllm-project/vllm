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

from .index import prepare_chunk_indices
from .op import exp
from .utils import FLA_CHUNK_SIZE, autotune_unless_cdna3, is_cdna3

_KKT_STOCK = [
    triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
    for BK in [32, 64, 128]
    for num_warps in [2, 4, 8]
    for num_stages in [2, 3, 4]
]

_KKT_CDNA3 = (
    triton.Config(
        {"BK": 32, "matrix_instr_nonkdim": 16, "waves_per_eu": 0},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BK": 128, "matrix_instr_nonkdim": 16, "waves_per_eu": 0},
        num_warps=4,
        num_stages=1,
    ),
)


def _kkt_cdna3_config(tokens_heads: int) -> triton.Config:
    small, large = _KKT_CDNA3
    return small if tokens_heads <= 65536 else large


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@autotune_unless_cdna3(configs=_KKT_STOCK, key=["H", "K", "BT", "IS_VARLEN"])
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
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
    m_t = o_t < T

    p_beta = beta + bos * H + i_h + o_t * H
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k = (
            k
            + (bos * Hg + i_h // (H // Hg)) * K
            + o_t[:, None] * (Hg * K)
            + o_k[None, :]
        )
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
        # Beta is applied after the loop, as in upstream FLA (see #65385).
        b_A = tl.dot(b_k, tl.trans(b_k), b_A)

    if USE_G:
        p_g = g + bos * H + i_h + o_t * H
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)
    b_A = b_A * b_beta[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = A + (bos * H + i_h) * BT + o_t[:, None] * (BT * H) + tl.arange(0, BT)[None, :]
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None])


def _chunk_kkt_warmup_inputs(
    *,
    dtype: torch.dtype,
    H: int,
    Hg: int,
    K: int,
    BT: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    # Straddles the threshold in _kkt_cdna3_config so both configs are compiled.
    tokens_heads: Any = WarmupChoices(1, 65537)
    return dict(
        k=TritonWarmupTensor(dtype, shape=(1, BT, Hg, K)),
        g=TritonWarmupTensor(torch.float32, shape=(1, BT, H)),
        beta=TritonWarmupTensor(dtype, shape=(1, BT, H)),
        A=TritonWarmupTensor(output_dtype, shape=(1, BT, H, BT)),
        cu_seqlens=TritonWarmupTensor(torch.int32, shape=(2,)),
        chunk_indices=TritonWarmupTensor(torch.int32, shape=(1, 2)),
        T=BT,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        num_chunks=1,
        num_batch_heads=H,
        tokens_heads=tokens_heads,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=chunk_scaled_dot_kkt_fwd_kernel,
    warmup_inputs=_chunk_kkt_warmup_inputs,
)
def _chunk_scaled_dot_kkt_launch(
    k: Any,
    g: Any,
    beta: Any,
    A: Any,
    cu_seqlens: Any,
    chunk_indices: Any,
    T: int,
    H: int,
    Hg: int,
    K: int,
    BT: int,
    num_chunks: int,
    num_batch_heads: int,
    tokens_heads: int,
) -> DispatchSpec:
    launch = {}
    if is_cdna3:
        launch = _kkt_cdna3_config(tokens_heads).all_kwargs()
    return (
        (num_chunks, num_batch_heads),
        dict(
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
            **launch,
        ),
        A,
    )


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_indices (torch.Tensor):
            Pre-computed chunk indices. If None and cu_seqlens is provided,
            computed internally. Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    return _chunk_scaled_dot_kkt_launch(
        k=k,
        g=g,
        beta=beta,
        A=torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        num_chunks=NT,
        num_batch_heads=B * H,
        tokens_heads=B * T * H,
    )
