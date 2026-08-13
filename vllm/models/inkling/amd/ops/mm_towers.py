# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused CUDA kernels for the Inkling vision/audio towers.

Both kernels keep the reference paths' fp32 accumulation and per-op bf16
rounding points (native ``rms_norm`` / ``F.gelu``); outputs are frequently
bit-identical and otherwise differ by 1-2 bf16 ulps from reduction-order
(real-checkpoint-weight cosine vs reference > 0.9999998).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton

from .norm import _get_num_warps_from_block_size


@triton.jit
def _dmel_embed_sum_norm_kernel(
    idx_ptr,  # [T, NB] int32 dMel bin indices (values in [0, VOCAB))
    w_ptr,  # [NB * VOCAB, D] bf16 embedding table
    norm_w_ptr,  # [D] (unused if HAS_NORM=False)
    out_ptr,  # [T, D] bf16
    eps,
    D,
    stride_idx_t,
    NB: tl.constexpr,
    VOCAB: tl.constexpr,
    D_P2: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, D_P2)
    mask = offs < D
    # One embedding row per mel bin (bin b uses table rows [b*VOCAB, (b+1)*VOCAB)),
    # summed in fp32 (matches torch's fp32-accumulated bf16 .sum()).
    acc = tl.zeros([D_P2], dtype=tl.float32)
    for b in tl.static_range(NB):
        v = tl.load(idx_ptr + t * stride_idx_t + b)
        row = (b * VOCAB + v).to(tl.int64)
        acc += tl.load(w_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    h = acc.to(tl.bfloat16)
    if HAS_NORM:
        # Match ir.ops.rms_norm: fp32 variance/normalize, then a single-rounded
        # bf16 multiply with the bf16 weight.
        x32 = h.to(tl.float32)
        var = tl.sum(tl.where(mask, x32 * x32, 0.0), axis=0) / D
        xn = (x32 * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
        w = tl.load(norm_w_ptr + offs, mask=mask, other=0.0)
        h = xn * w
    tl.store(out_ptr + t * D + offs, h, mask=mask)


def dmel_embed_sum_norm(
    dmel_idx: torch.Tensor,  # [T, NB] int32
    weight: torch.Tensor,  # [NB * VOCAB, D] bf16
    norm_weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """``rmsnorm(sum_b weight[b * VOCAB + idx[:, b]])`` in one launch (no
    [T, NB, D] intermediate)."""
    T, nb = dmel_idx.shape
    D = weight.shape[1]
    assert weight.shape[0] % nb == 0
    vocab = weight.shape[0] // nb
    out = torch.empty((T, D), dtype=weight.dtype, device=weight.device)
    if T == 0:
        return out
    d_p2 = triton.next_power_of_2(D)
    _dmel_embed_sum_norm_kernel[(T,)](
        dmel_idx,
        weight,
        norm_weight if norm_weight is not None else weight,
        out,
        eps,
        D,
        dmel_idx.stride(0),
        NB=nb,
        VOCAB=vocab,
        D_P2=d_p2,
        HAS_NORM=norm_weight is not None,
        # Swept on GB200: 8 warps beats the block-size heuristic's 16 by ~4%
        # at large T (the 80-row gather chain is latency- not lane-bound).
        num_warps=8,
    )
    return out


@triton.jit
def _rmsnorm_gelu_kernel(
    x_ptr,  # [R, D] bf16
    w_ptr,  # [D]
    out_ptr,  # [R, D] bf16 (or the folded layout when FOLD)
    eps,
    R,
    D,
    D_P2: tl.constexpr,
    BLOCK_M: tl.constexpr,  # rows per block (>1 for small D)
    HAS_GELU: tl.constexpr,
    FOLD: tl.constexpr,
    # fold geometry: input rows index [N, T, H, W]; the store scatters each
    # row to (out_row, slot) of fold_timespace_to_depth's output layout.
    FT: tl.constexpr,
    FH: tl.constexpr,
    FW: tl.constexpr,
    TF: tl.constexpr,
    HF: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < R
    offs = tl.arange(0, D_P2)
    mask = rmask[:, None] & (offs < D)[None, :]
    x32 = tl.load(x_ptr + rows[:, None] * D + offs[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    var = tl.sum(x32 * x32, axis=1) / D
    xn = (x32 * tl.math.rsqrt(var + eps)[:, None]).to(tl.bfloat16)
    w = tl.load(w_ptr + offs, mask=offs < D, other=0.0)
    h = xn * w[None, :]  # bf16 multiply, matching ir.ops.rms_norm
    if HAS_GELU:
        # Exact (erf) GELU on the bf16-rounded norm output, fp32 math,
        # matching F.gelu's opmath on a bf16 tensor.
        g32 = h.to(tl.float32)
        h = (0.5 * g32 * (1.0 + tldevice.erf(g32 * 0.7071067811865476))).to(tl.bfloat16)
    if FOLD:
        # Store directly into the next layer's folded layout (a pure
        # permutation — replaces the separate fold copy pass).
        t = (rows // (FH * FW)) % FT
        hh = (rows // FW) % FH
        ww = rows % FW
        n = rows // (FT * FH * FW)
        slot = ((t % TF) * HF + hh % HF) * HF + ww % HF
        out_row = ((n * (FT // TF) + t // TF) * (FH // HF) + hh // HF) * (
            FW // HF
        ) + ww // HF
        base = (out_row * (TF * HF * HF) + slot) * D
        tl.store(out_ptr + base[:, None] + offs[None, :], h, mask=mask)
    else:
        tl.store(out_ptr + rows[:, None] * D + offs[None, :], h, mask=mask)


def rmsnorm_gelu(
    x: torch.Tensor,  # [..., D] bf16 contiguous
    weight: torch.Tensor,
    eps: float,
    gelu: bool = True,
    fold: tuple[int, int] | None = None,  # (t_fold, hw_fold) of the NEXT fold
) -> torch.Tensor:
    """Fused ``gelu(rmsnorm(x))`` (or plain rmsnorm); multiple rows per block
    when D is small. With ``fold``, x must be [N, T, H, W, D] and the output
    comes back as ``fold_timespace_to_depth(result, *fold)``."""
    D = x.shape[-1]
    flat = x.reshape(-1, D)
    assert flat.stride(1) == 1 and flat.stride(0) == D
    R = flat.shape[0]
    if fold is None:
        out = torch.empty_like(flat)
        ft = fh = fw = tf = hf = 1
        out_shape = x.shape
    else:
        tf, hf = fold
        N, ft, fh, fw, _ = x.shape
        out_shape = (N, ft // tf, fh // hf, fw // hf, tf * hf * hf * D)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if R == 0:
        return out.reshape(out_shape)
    d_p2 = triton.next_power_of_2(D)
    block_m = max(1, 4096 // d_p2)
    _rmsnorm_gelu_kernel[(triton.cdiv(R, block_m),)](
        flat,
        weight,
        out,
        eps,
        R,
        D,
        D_P2=d_p2,
        BLOCK_M=block_m,
        HAS_GELU=gelu,
        FOLD=fold is not None,
        FT=ft,
        FH=fh,
        FW=fw,
        TF=tf,
        HF=hf,
        num_warps=_get_num_warps_from_block_size(d_p2 * block_m),
    )
    return out.reshape(out_shape)
