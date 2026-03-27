# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import os

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .wy_fast import recompute_w_u_fwd

FLA_TRIL_PRECISION = os.environ.get("FLA_TRIL_PRECISION", "ieee")


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """
    Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass.

    This kernel fuses chunk_scaled_dot_kkt_fwd and solve_tril into a single kernel,
    avoiding the HBM round-trip for the intermediate A matrix.

    Steps:
    1. Compute all 10 lower-triangular [BC, BC] blocks of beta * K @ K^T in registers
    2. Apply gate and beta scaling
    3. Forward substitution on diagonal blocks
    4. Block merge to get full (I+A)^{-1}
    5. Write result to A (output)
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    # GQA: key head index
    i_hg = i_h // (H // Hg)

    k += (bos * Hg + i_hg) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # load beta for each sub-chunk
    p_b0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    # load gate if used
    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))

        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    ############################################################################
    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    ############################################################################

    # 4 diagonal blocks
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    # 6 off-diagonal blocks
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(
            k, (T, K), (Hg * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        # diagonal block 0
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            # diagonal block 1
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            # off-diagonal (1,0)
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                # diagonal block 2
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                # off-diagonal (2,0), (2,1)
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (Hg * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    # diagonal block 3
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    # off-diagonal (3,0), (3,1), (3,2)
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    ############################################################################
    # Step 2: apply gate and beta scaling
    ############################################################################

    if USE_G:
        # diagonal blocks: g_diff = g_i - g_j within sub-chunk
        b_A00 *= exp(b_g0[:, None] - b_g0[None, :])
        b_A11 *= exp(b_g1[:, None] - b_g1[None, :])
        b_A22 *= exp(b_g2[:, None] - b_g2[None, :])
        b_A33 *= exp(b_g3[:, None] - b_g3[None, :])

        # off-diagonal blocks: g_diff = g_row - g_col (cross sub-chunk)
        b_A10 *= exp(b_g1[:, None] - b_g0[None, :])
        b_A20 *= exp(b_g2[:, None] - b_g0[None, :])
        b_A21 *= exp(b_g2[:, None] - b_g1[None, :])
        b_A30 *= exp(b_g3[:, None] - b_g0[None, :])
        b_A31 *= exp(b_g3[:, None] - b_g1[None, :])
        b_A32 *= exp(b_g3[:, None] - b_g2[None, :])

    # apply beta to row dimension and mask
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # diagonal blocks: strictly lower triangular within sub-chunk, scaled by beta
    b_A00 = (
        tl.where(m_d & (m_tc0[:, None] & m_tc0[None, :]), b_A00, 0.0) * b_b0[:, None]
    )
    b_A11 = (
        tl.where(m_d & (m_tc1[:, None] & m_tc1[None, :]), b_A11, 0.0) * b_b1[:, None]
    )
    b_A22 = (
        tl.where(m_d & (m_tc2[:, None] & m_tc2[None, :]), b_A22, 0.0) * b_b2[:, None]
    )
    b_A33 = (
        tl.where(m_d & (m_tc3[:, None] & m_tc3[None, :]), b_A33, 0.0) * b_b3[:, None]
    )

    # off-diagonal blocks: full block, scaled by beta
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    ############################################################################
    # Step 3: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    #
    # Same algorithm as solve_tril, but rows are extracted from in-register
    # [BC, BC] tensor via tl.sum(tl.where(mask, tensor, 0), 0) instead of
    # tl.load from HBM.
    ############################################################################

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ############################################################################
    # Step 4: block merge -> full (I + A)^{-1}
    ############################################################################

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=DOT_PRECISION),
        b_Ai00,
        input_precision=DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=DOT_PRECISION),
        b_Ai11,
        input_precision=DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=DOT_PRECISION),
        b_Ai22,
        input_precision=DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=DOT_PRECISION)
        + tl.dot(b_A21, b_Ai10, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=DOT_PRECISION)
        + tl.dot(b_A32, b_Ai21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=DOT_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=DOT_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    ############################################################################
    # Step 5: store full (I + A)^{-1} to output A
    ############################################################################

    p_A00 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_A30 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_A33 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    GDN intra-chunk forward: fused kkt + solve_tril + recompute_w_u.

    Equivalent to:
        A = chunk_scaled_dot_kkt_fwd(k, g, beta, ...)       # kernel 1
        A = solve_tril(A, ...)                                # kernel 2
        w, u = recompute_w_u_fwd(k, v, beta, A, g, ...)      # kernel 3

    Fuses kernels 1+2 into a single kernel, reducing from 3 to 2 kernel launches
    and eliminating the HBM round-trip for the intermediate A matrix.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, Hg, K]`.
        v (torch.Tensor):
            The value tensor of shape `[B, T, H, V]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths. Default: `None`.
        chunk_size (int):
            The chunk size. Default: 64.

    Returns:
        w (torch.Tensor): shape `[B, T, H, K]`
        u (torch.Tensor): shape `[B, T, H, V]`
        A (torch.Tensor): shape `[B, T, H, BT]`, the solved (I+A)^{-1} matrix
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    BC = 16

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Step 1: fused kkt + solve_tril
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=BC,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )

    # Step 2: recompute_w_u
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    return w, u, A
