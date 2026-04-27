# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/chunk.py
# (only the forward cumsum helper is vendored — the rest of rwkv6/chunk.py
# is not used by RWKV-7 inference).
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from ..index import prepare_chunk_indices
from ..utils import use_cuda_graph


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps, num_stages=num_stages)
        for BS in [16, 32, 64]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["S", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_rwkv6_fwd_cumsum_kernel(
    s,
    oi,
    oe,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1.0, 0.0).to(tl.float32)

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_oi = tl.make_block_ptr(
        oi + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_oe = tl.make_block_ptr(
        oe + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_oi = tl.dot(m_i, b_s)
    b_oe = tl.dot(m_e, b_s)
    tl.store(
        p_oi,
        b_oi.to(p_oi.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_oe,
        b_oe.to(p_oe.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chunk_rwkv6_fwd_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative decay (inclusive + exclusive) over chunks of size BT.

    ``g`` has shape ``[B, T, H, S]``; returns ``(gi, ge)`` of the same shape
    in fp32, where ``gi[i] = sum_{j <= i} g[j]`` (inclusive) and
    ``ge[i] = sum_{j < i} g[j]`` (exclusive) within each chunk.
    """
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    gi = torch.empty_like(g, dtype=torch.float32)
    ge = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g,
        gi,
        ge,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        S=S,
        BT=BT,
    )
    return gi, ge
