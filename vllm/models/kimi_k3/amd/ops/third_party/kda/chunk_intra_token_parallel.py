# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Forward-only adaptation of flash-linear-attention 0.5.0.
# ruff: noqa: E501

# Token-parallel implementation of KDA intra chunk kernel

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp2
from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H", "HV"],
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N

        # Unrolled binary search (max B=2^32)
        # We can limit iterations based on expected max batch size if needed
        # 20 iterations covers B=1M, usually enough
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    G: tl.constexpr = HV // H

    q += bos * H * K
    k += bos * H * K
    g += bos * HV * K
    Aqk += bos * HV * BT
    Akk += bos * HV * BC
    beta += bos * HV

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    o_k = tl.arange(0, BK)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    # q/k: [B, T, H, K], manual load via mapped qk head index
    p_qk = o_h[:, None] * K + o_k[None, :]
    b_q = tl.load(q + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k = tl.load(k + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)

    # g: [B, T, HV, K], beta: [B, T, HV]
    p_g = tl.make_block_ptr(
        g + i_t * HV * K, (HV, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_beta = tl.make_block_ptr(beta + i_t * HV, (HV,), (1,), (i_hg * BH,), (BH,), (0,))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_k *= b_beta[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        b_kj = tl.load(k + j * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
        p_gj = tl.make_block_ptr(
            g + j * HV * K, (HV, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
        )
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(
            Aqk + i_t * HV * BT + o_hv * BT + j % BT,
            b_Aqk.to(Aqk.dtype.element_ty),
            mask=m_hv,
        )
        tl.store(
            Akk + i_t * HV * BC + o_hv * BC + j - i_ts,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_hv,
        )


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    """
    Token-parallel implementation: each token gets its own thread block.
    Supports both fixed-length and variable-length sequences.
    Reduces wasted computation on padding.

    Writes directly to Aqk and Akk tensors (in-place).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        gk: [B, T, HV, K] cumsum of gates (HV >= H for GVA)
        beta: [B, T, HV]
        Aqk: [B, T, HV, BT] output tensor to write to
        Akk: [B, T, HV, BC] output tensor for diagonal blocks (fp32)
        scale: attention scale
        chunk_size: BT (default 64)
        sub_chunk_size: BC (default 16)
    """
    B, T, H, K, HV = *q.shape, gk.shape[2]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta):
        return (B * T, triton.cdiv(HV, meta["BH"]))

    chunk_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk
