# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 per-token activation quantization ("fouroversix_mae" scale rule).

Each row gets a per-token global scale `amax / 1536`. For every 16-element
block both a 6.0- and a 4.0-based e4m3 block scale are tried and the one with
the smaller absolute reconstruction error is kept.
"""

import math

import torch

from vllm.triton_utils import tl, triton

from .block_scales import _scale_offsets

FP32_MAX = tl.constexpr(3.4028234663852886e38)


@triton.jit
def _pack_roundtrip(x):
    low, high = tl.split(x)
    packed, halves = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
        "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
        "=r,=r,f,f",
        [low, high],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    low = (halves & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    high = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return packed.to(tl.uint8), tl.join(low, high).to(tl.float32)


@triton.jit
def _block_error(x, reconstructed):
    groups: tl.constexpr = x.shape[0]
    error = tl.abs(reconstructed - x)
    for width in tl.static_range(3, -1, -1):
        low, high = tl.split(tl.reshape(error, (groups, 1 << width, 2)))
        error = low + high
    return tl.reshape(error, (groups,))


@triton.jit
def _block_scales(block_max, encoder_scale, decoder_scale):
    """The 6.0- and 4.0-based block scales of blocks with max |x| `block_max`
    and their inverses."""
    scale6 = tl.minimum(block_max * (encoder_scale * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(block_max * (encoder_scale * 0.25), 448.0)
    scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
    scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder_scale), FP32_MAX)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder_scale), FP32_MAX)
    return scale6, scale4, inverse6, inverse4


@triton.jit
def _quantize_block(x, encoder_scale, decoder_scale, outer_scale, BLOCK: tl.constexpr):
    x = tl.reshape(x, (BLOCK // 16, 16))
    scale6, scale4, inverse6, inverse4 = _block_scales(
        tl.max(tl.abs(x), 1), encoder_scale, decoder_scale
    )
    packed6, rounded6 = _pack_roundtrip(
        tl.reshape(x * inverse6[:, None], (BLOCK // 16, 8, 2))
    )
    packed4, rounded4 = _pack_roundtrip(
        tl.reshape(x * inverse4[:, None], (BLOCK // 16, 8, 2))
    )
    reconstructed6 = tl.reshape(rounded6, (BLOCK // 16, 16)) * scale6[:, None]
    reconstructed4 = tl.reshape(rounded4, (BLOCK // 16, 16)) * scale4[:, None]
    error6 = _block_error(x, reconstructed6 * outer_scale)
    error4 = _block_error(x, reconstructed4 * outer_scale)
    choose4 = error4 < error6
    packed = tl.reshape(tl.where(choose4[:, None], packed4, packed6), (BLOCK // 2,))
    scales = tl.where(choose4, scale4, scale6).to(tl.float8e4nv)
    return packed, scales


@triton.jit
def _e2m1_pair(a, b):
    """The e2m1x2 byte of `(a, b)` (`a` in the low nibble) and both
    values rounded to e2m1."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 q; .reg .b32 h; .reg .b16 l, u; "
        "cvt.rn.satfinite.e2m1x2.f32 q, $4, $3; cvt.u32.u8 $0, q; "
        "cvt.rn.f16x2.e2m1x2 h, q; mov.b32 {l, u}, h; "
        "cvt.f32.f16 $1, l; cvt.f32.f16 $2, u; }",
        "=r,=f,=f,f,f",
        [a, b],
        dtype=(tl.uint32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair(x, k: tl.constexpr, inverse, scale, outer_scale):
    """Columns `2k` and `2k + 1` of `_candidate`: their e2m1 byte at
    byte `k` and their reconstruction error."""
    byte, low, high = _e2m1_pair(x[2 * k] * inverse, x[2 * k + 1] * inverse)
    error = tl.abs((low * scale) * outer_scale - x[2 * k]) + tl.abs(
        (high * scale) * outer_scale - x[2 * k + 1]
    )
    return byte.to(tl.uint64) << (8 * k), error


@triton.jit
def _quad(x, k: tl.constexpr, inverse, scale, outer_scale):
    low, low_error = _pair(x, 2 * k, inverse, scale, outer_scale)
    high, high_error = _pair(x, 2 * k + 1, inverse, scale, outer_scale)
    return low | high, low_error + high_error


@triton.jit
def _octet(x, k: tl.constexpr, inverse, scale, outer_scale):
    low, low_error = _quad(x, 2 * k, inverse, scale, outer_scale)
    high, high_error = _quad(x, 2 * k + 1, inverse, scale, outer_scale)
    return low | high, low_error + high_error


@triton.jit
def _candidate(x, inverse, scale, outer_scale):
    """Packed values (uint64) and the reconstruction error of the blocks
    `x` (16 columns) with block scale `scale`: the error is summed
    pairwise, as in `_block_error`."""
    low, low_error = _octet(x, 0, inverse, scale, outer_scale)
    high, high_error = _octet(x, 1, inverse, scale, outer_scale)
    return low | high, low_error + high_error


@triton.jit
def _quantize_columns(x, encoder_scale, decoder_scale, outer_scale):
    """`_quantize_block` of blocks given as a tuple `x` of their 16 fp32
    columns, for each thread to hold whole blocks: packed values (uint64) and
    e4m3 scales."""
    block_max = tl.abs(x[0])
    for j in tl.static_range(1, 16):
        block_max = tl.maximum(block_max, tl.abs(x[j]))
    scale6, scale4, inverse6, inverse4 = _block_scales(
        block_max, encoder_scale, decoder_scale
    )
    packed6, error6 = _candidate(x, inverse6, scale6, outer_scale)
    packed4, error4 = _candidate(x, inverse4, scale4, outer_scale)
    choose4 = error4 < error6
    return (
        tl.where(choose4, packed4, packed6),
        tl.where(choose4, scale4, scale6).to(tl.float8e4nv),
    )


@triton.jit
def _store_block(
    Values,
    Scales,
    row,
    start,
    packed,
    scales,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    # Stores the block at column ``start`` of output row ``row``.
    tl.store(Values + row * (K // 2) + start // 2 + tl.arange(0, BLOCK // 2), packed)
    scale_cols = start // 16 + tl.arange(0, BLOCK // 16)
    tl.store(Scales + _scale_offsets(row, scale_cols, K // 16, TILE), scales)


@triton.jit
def _quantize_row(X, Values, Scales, Amax, row, K: tl.constexpr, BLOCK: tl.constexpr):
    """Quantizes row `row` of `X`, with row-major scales and the row amax
    in `Amax`."""
    row = row.to(tl.int64)
    amax = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, K, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * K + cols, cols < K, other=0)
        amax = tl.maximum(amax, tl.abs(x).to(tl.float32))
    amax = tl.max(amax, 0)
    tl.store(Amax + row, amax)
    encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
    decoder_scale = tl.div_rn(1.0, encoder_scale)
    outer_scale = amax * (1.0 / 1536.0)
    for start in range(0, K, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
        packed, scales = _quantize_block(
            x, encoder_scale, decoder_scale, outer_scale, BLOCK
        )
        _store_block(Values, Scales, row, start, packed, scales, K, BLOCK, 0)


@triton.jit
def _nvfp4_quantize_kernel(
    X, Values, Scales, Amax, M, K: tl.constexpr, BLOCK: tl.constexpr
):
    tl.extra.cuda.gdc_wait()
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        _quantize_row(X, Values, Scales, Amax, row, K, BLOCK)


def nvfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize rows of bf16 `x` [M, K] to NVFP4.

    Returns packed e2m1 values, row-major e4m3 block scales and the fp32
    per-row amax.
    """
    assert x.ndim == 2 and x.is_contiguous() and x.shape[1] % 16 == 0
    num_rows, k = x.shape
    values = torch.empty((num_rows, k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty(
        (num_rows, k // 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    amax = torch.empty(num_rows, dtype=torch.float32, device=x.device)
    # Wide programs for latency with few rows, narrow ones for occupancy.
    block, num_warps = (512, 4) if num_rows <= 512 else (256, 1)
    block = math.gcd(k, block)
    _nvfp4_quantize_kernel[(min(num_rows, 8192),)](
        x,
        values,
        scales,
        amax,
        num_rows,
        K=k,
        BLOCK=block,
        num_warps=num_warps,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return values, scales, amax
