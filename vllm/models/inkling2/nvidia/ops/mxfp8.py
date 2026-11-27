# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 (e4m3 values, ue8m0 scales per 32 elements) activation quantization."""

import math

import torch

from vllm.triton_utils import tl, tldevice, triton

from .block_scales import _scale_offsets


@triton.jit
def _mxfp8_block(groups):
    amax = tl.max(tl.abs(groups), 1)
    raw_scale = amax * (1.0 / 448.0)
    min_scale = tl.full((), 2.0**-127, tl.float32)
    exponent = ((raw_scale.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = tl.where(raw_scale <= min_scale, 0, tl.minimum(exponent, 254))
    exponent = tl.where(raw_scale != raw_scale, 255, exponent)
    scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), min_scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    inverse = tl.where(exponent == 0, 2.0**127, tldevice.rcp_rn(scale))
    inverse = tl.where(exponent == 255, 0.0, inverse)
    values = groups * inverse[:, None]
    values = tl.minimum(
        tl.maximum(values, -448.0), 448.0, propagate_nan=tl.PropagateNan.ALL
    )
    return values.to(tl.float8e4nv), exponent


@triton.jit
def _store_mxfp8_block(
    Values,
    Scales,
    row,
    start,
    values,
    exponent,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    # Stores the block at column ``start`` of output row ``row``.
    groups = start // 32 + tl.arange(0, BLOCK // 32)
    tl.store(Scales + _scale_offsets(row, groups, N // 32, TILE), exponent)
    cols = start + tl.arange(0, BLOCK)
    tl.store(Values + row * N + cols, tl.reshape(values, (BLOCK,)))


@triton.jit
def _mxfp8_quantize_kernel(X, Y, S, M, N: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    for r in range(tl.program_id(0), M, tl.num_programs(0)):
        row = tl.cast(r, tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
        values, exponent = _mxfp8_block(tl.reshape(x, (BLOCK // 32, 32)))
        _store_mxfp8_block(
            Y, S, row, tl.program_id(1) * BLOCK, values, exponent, N, BLOCK, 0
        )


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows of bf16 `x` [M, N] to MXFP8 with round-up power-of-two scales.

    Returns e4m3 values and row-major ue8m0 scale exponents [M, N/32].
    """
    assert x.ndim == 2 and x.is_contiguous() and x.shape[1] % 32 == 0
    rows, n = x.shape
    values = torch.empty((rows, n), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, n // 32), dtype=torch.uint8, device=x.device)
    block = math.gcd(n, 1024)
    _mxfp8_quantize_kernel[(min(rows, 8192), n // block)](
        x,
        values,
        scales,
        rows,
        N=n,
        BLOCK=block,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return values, scales
