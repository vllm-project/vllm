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
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py
# Forward path only.
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from ..index import prepare_chunk_indices
from ..utils import check_shared_mem, is_amd, use_cuda_graph

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]
BK_LIST = [32, 64, 128] if check_shared_mem() else [16, 32]


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BK_LIST
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_kernel_o(
    qg,
    v,
    v_new,
    A_qk,
    A_qb,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_qg = tl.make_block_ptr(
            qg + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, b_h)

    p_Aqk = tl.make_block_ptr(
        A_qk + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    p_Aqb = tl.make_block_ptr(
        A_qb + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_v_new = tl.make_block_ptr(
        v_new + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
    b_Aqb = tl.load(p_Aqb, boundary_check=(0, 1))
    b_Aqk = tl.where(m_s, b_Aqk, 0)
    b_Aqb = tl.where(m_s, b_Aqb, 0)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
    b_o = (
        b_o
        + tl.dot(b_Aqk.to(b_v.dtype), b_v)
        + tl.dot(b_Aqb.to(b_v_new.dtype), b_v_new)
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_dplr_fwd_o(
    qg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    h: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_dplr_fwd_kernel_o[grid](
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o
