# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 fused MoE tail (SM100, TP1): top-k reduce of the deferred GEMM2 rows
(fp32, k order, no atomics) + shared-expert add, optionally fused with the next
layer's residual add + Gemma RMSNorm + MXFP8 quant. Rounds once per output."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

_BLOCK_H = 2048  # best on GB300 for both paths at every M in 1-32k


@triton.jit
def _moe_finalize_kernel(
    perm_ptr,  # [R, H] bf16 unweighted GEMM2 rows
    w_ptr,  # [M, K] route scales
    idx_ptr,  # [M, K] int32 row map, -1 = skip
    shared_ptr,  # [M, H] bf16 or None
    resid_ptr,  # [M, H] bf16 (norm/quant only)
    normw_ptr,  # [H] bf16 Gemma norm weight (norm/quant only)
    out_ptr,  # [M, H] bf16: finalized output, or the updated residual with norm/quant
    q_ptr,  # [M, H] e4m3 (norm/quant only)
    sf_ptr,  # 128x4 swizzled UE8M0 scales (norm/quant only)
    eps,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    RUN_NORM_QUANT: tl.constexpr,
):
    m = tl.program_id(0)
    tl.extra.cuda.gdc_wait()

    # Top-k reduce (fp32, k order) + shared add (+ residual add). The norm
    # needs the whole row, so keep the chunks resident (a tuple, since a plain
    # annotation would be resolved as a global by the Triton parser).
    NCHUNK: tl.constexpr = H // BLOCK_H
    hs = ()
    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK_H + tl.arange(0, BLOCK_H)
        h = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for k in tl.static_range(K):
            row = tl.load(idx_ptr + m * K + k)
            wk = tl.load(w_ptr + m * K + k).to(tl.float32)
            x = tl.load(perm_ptr + tl.maximum(row, 0) * H + offs).to(tl.float32)
            h = tl.fma(wk, tl.where(row >= 0, x, 0.0), h)

        if shared_ptr is not None:
            h += tl.load(shared_ptr + m * H + offs).to(tl.float32)
        if RUN_NORM_QUANT:
            h += tl.load(resid_ptr + m * H + offs).to(tl.float32)
        tl.store(out_ptr + m * H + offs, h.to(tl.bfloat16))
        hs = hs + (h,)  # type: ignore[assignment]

    if RUN_NORM_QUANT:
        # RMSNorm
        ssq = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for c in tl.static_range(NCHUNK):
            ssq += hs[c] * hs[c]
        rstd = tl.rsqrt(tl.sum(ssq[None, :], axis=1) / H + eps)

        # MXFP8 quant, one scale per 32 elements
        SFB: tl.constexpr = BLOCK_H // 32
        g = tl.arange(0, SFB)
        inv448: tl.constexpr = 0.002232142857142857  # rn(1/448) float32
        for c in tl.static_range(NCHUNK):
            offs = c * BLOCK_H + tl.arange(0, BLOCK_H)
            wv = tl.load(normw_ptr + offs).to(tl.float32)
            y = ((hs[c] * rstd) * (wv + 1.0)).to(tl.bfloat16).to(tl.float32)
            amax = tl.max(tl.abs(tl.reshape(y, (SFB, 32))), axis=1)

            # UE8M0 = ceil exponent of amax / 448 (float_to_ue8m0)
            bits = (amax * inv448).to(tl.uint32, bitcast=True)
            exp = bits >> 23
            mant = bits & 0x7FFFFF
            bump = (mant != 0) & ((exp != 0) | (mant > 0x400000))
            e = tl.minimum(exp + bump.to(tl.uint32), 254)

            # 1 / scale = 2^(127 - e), built as a bf16 exponent
            inv = ((254 - e) << 7).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            inv = tl.where(e == 0, 0.0, inv.to(tl.float32))
            inv = tl.reshape(tl.broadcast_to(inv[:, None], (SFB, 32)), (BLOCK_H,))
            qf = tl.minimum(y * inv, 448.0)
            tl.store(q_ptr + m * H + offs, qf.to(tl.float8e4nv))

            # 128x4 swizzled scale layout; rows >= M of the last 128-row tile
            # are left unwritten (MXFP8 GEMMs never read them into stored
            # outputs).
            col = c * SFB + g
            sf_offs = (
                (col % 4)
                + (col // 4) * 512
                + (m % 32) * 16
                + ((m % 128) // 32) * 4
                + (m // 128) * (128 * (H // 32))
            )
            tl.store(sf_ptr + sf_offs, e.to(tl.uint8))


def moe_finalize_add(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared: torch.Tensor | None,
) -> torch.Tensor:
    """Top-k reduce of deferred GEMM2 rows (an ``UnfinalizedMoEOutput``) plus the
    shared-expert output, as a bf16 [num_tokens, H] tensor."""
    num_tokens, top_k = expanded_idx_to_permuted_idx.shape
    hidden_dim = gemm2_permuted.shape[1]
    assert hidden_dim % _BLOCK_H == 0, hidden_dim
    assert gemm2_permuted.stride(1) == 1
    assert shared is None or shared.is_contiguous()
    out = gemm2_permuted.new_empty((num_tokens, hidden_dim))
    if num_tokens == 0:
        return out
    _moe_finalize_kernel[(num_tokens,)](
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        shared,
        None,
        None,
        out,
        None,
        None,
        0.0,
        H=hidden_dim,
        K=top_k,
        BLOCK_H=_BLOCK_H,
        RUN_NORM_QUANT=False,
        num_warps=8,
        launch_pdl=True,
    )
    return out


def moe_finalize_add_norm_mxfp8(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared: torch.Tensor | None,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``moe_finalize_add`` fused with the next layer's
    ``gemma_fused_add_rmsnorm_mxfp8``.

    Returns:
        (updated residual [M, H] bf16, e4m3 [M, H], 128x4 swizzled UE8M0 scales
        with the padding rows left unwritten).

    """
    num_tokens, top_k = expanded_idx_to_permuted_idx.shape
    hidden_dim = gemm2_permuted.shape[1]
    assert hidden_dim % _BLOCK_H == 0, hidden_dim
    assert residual.shape == (num_tokens, hidden_dim) and residual.is_contiguous()
    resid_out = torch.empty_like(residual)
    q = residual.new_empty((num_tokens, hidden_dim), dtype=torch.float8_e4m3fn)
    sf = residual.new_empty(
        round_up(num_tokens, 128) * (hidden_dim // 32), dtype=torch.uint8
    )
    if num_tokens == 0:
        return resid_out, q, sf
    _moe_finalize_kernel[(num_tokens,)](
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        shared,
        residual,
        norm_weight,
        resid_out,
        q,
        sf,
        eps,
        H=hidden_dim,
        K=top_k,
        BLOCK_H=_BLOCK_H,
        RUN_NORM_QUANT=True,
        num_warps=8,
        launch_pdl=True,
    )
    return resid_out, q, sf
