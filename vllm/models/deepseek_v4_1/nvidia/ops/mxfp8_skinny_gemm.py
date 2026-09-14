# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton SIMT skinny GEMM for DeepSeek-V4.1 MXFP8/BF16 projections (M <= 4).

Computes ``Y[M, N] = X[M, K] @ W[N, K].T`` with simple data-parallel threads;
each program covers ``BN`` output columns and statically unrolls ``BM`` rows.
Accepts either a BF16 weight (plain dot product) or an MXFP8 weight whose
per-32 E8M0 scales are in the F8_128x4 swizzled layout produced by
:func:`swizzle_mxfp8_scale`.

BF16 activations are quantized in-kernel with ``round_mx``: per 32-element
block, amax is scaled by 1/448 and rounded up to an E8M0 power of two. This is
the same math as ``mxfp8_e4m3_quantize`` (and, with 32-element groups, as
``fused_inv_rope_fp8_quant(quant_group_size=32)`` followed by a
``recipe=(1, 1, 32)`` fp8_einsum), so the output matches the production MXFP8
baseline over the swept input scales 1e-5..100 (rel-RMSE <= 3e-4 measured on
B200). The quantization is recomputed per output tile; at these sizes the
redundancy is cheaper than a separate quantization kernel.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    swizzle_mxfp8_scale,
)
from vllm.triton_utils import tl, triton


@dataclass(frozen=True, slots=True)
class Mxfp8SimtGemmConfig:
    bm: int
    bn: int
    warps: int = 4


@triton.jit
def _sf_offset(row, block_k, K: tl.constexpr):
    return (
        ((row // 128 * tl.cdiv(K, 128) + block_k // 4) * 32 + row % 32) * 16
        + row % 128 // 32 * 4
        + block_k % 4
    )


@triton.jit
def _decode_scale(sf):
    # E8M0 exponents map directly to FP32 exponent bits except code zero,
    # which represents the subnormal value 2**-127.
    return tl.where(
        sf == 0,
        5.877471754111438e-39,
        (sf.to(tl.uint32) << 23).to(tl.float32, bitcast=True),
    )


@triton.jit
def _round_mx(a, ROWS: tl.constexpr, BK: tl.constexpr):
    grouped = tl.reshape(a, (ROWS, BK // 32, 32))
    normalized = tl.max(tl.abs(grouped), 2) * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    inv = tl.where(sf == 0, 0, (254 - sf) << 23).to(tl.float32, bitcast=True)
    aq = (grouped * inv[:, :, None]).to(tl.float8e4nv).to(tl.float32)
    return tl.reshape(aq * _decode_scale(sf)[:, :, None], (ROWS, BK))


@triton.jit
def _mxfp8_simt_kernel(
    A,
    W,
    SW,
    SA,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    MX: tl.constexpr,
    PRE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ni = tl.program_id(0) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    w = tl.load(
        W + ni[:, None] * K + ki[None, :], (ni[:, None] < N) & (ki[None, :] < K), 0.0
    ).to(tl.float32)
    if MX:
        sw = tl.load(
            SW + _sf_offset(ni[:, None], ki[None, :] // 32, K),
            (ni[:, None] < N) & (ki[None, :] < K),
            127,
        ).to(tl.int32)
        w = w * _decode_scale(sw)
    for r in tl.static_range(BM):
        mi = tl.program_id(1) * BM + r
        a = tl.load(A + mi * AS + ki, (mi < M) & (ki < K), 0.0).to(tl.float32)
        if MX:
            if PRE:
                sa = tl.load(
                    SA + _sf_offset(mi, ki // 32, K), (mi < M) & (ki < K), 127
                ).to(tl.int32)
                a = a * _decode_scale(sa)
            else:
                a = tl.reshape(_round_mx(tl.reshape(a, (1, BK)), 1, BK), (BK,))
        y = tl.sum(w * a[None, :], 1)
        tl.store(Y + mi * N + ni, y, (mi < M) & (ni < N))


def mxfp8_simt_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor | None,
    sx: torch.Tensor | None,
    config: Mxfp8SimtGemmConfig,
    out_fp32: bool = False,
) -> torch.Tensor:
    """``x[M, K] @ w[N, K].T`` with the measured SIMT tile.

    ``sw``/``sx`` are F8_128x4-swizzled E8M0 scales for the MXFP8 operands;
    pass ``sw=None`` for a BF16 weight. ``sx=None`` re-quantizes BF16
    activations in-kernel (``_round_mx``), otherwise ``x`` must be FP8-E4M3
    with pre-quantized scales.
    """
    m, k = x.shape
    n = w.shape[0]
    assert x.stride(1) == 1 and w.is_contiguous()
    y = torch.empty(
        (m, n), device=x.device, dtype=torch.float32 if out_fp32 else torch.bfloat16
    )
    _mxfp8_simt_kernel[(triton.cdiv(n, config.bn), triton.cdiv(m, config.bm))](
        x,
        w,
        sw,
        sx,
        y,
        m,
        n,
        k,
        x.stride(0),
        sw is not None,
        sx is not None,
        config.bm,
        config.bn,
        triton.next_power_of_2(k),
        num_warps=config.warps,
    )
    return y


def swizzle_wo_a_packed_scale(packed: torch.Tensor) -> torch.Tensor:
    """One-shot wo_a weight-scale rearrangement for the SIMT kernel.

    ``packed`` is the per-row UE8M0 scale tensor stored for the DeepGEMM BMM
    path, shape ``(1024, 32)`` with 4 packed E8M0 bytes per element covering
    128 K elements. Unpacks to the linear ``(1024, 128)`` uint8 layout and
    returns the F8_128x4 swizzle the SIMT kernel reads.
    """
    n, k_scale = packed.shape
    shifts = torch.arange(4, device=packed.device) * 8
    linear = (
        ((packed.to(torch.int64)[:, :, None] >> shifts) & 255)
        .to(torch.uint8)
        .reshape(n, k_scale * 4)
    )
    return swizzle_mxfp8_scale(linear, n, k_scale * 128)
