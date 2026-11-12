# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slow references for the AMD monokernel I/O tests: plain torch in float64,
written from the model definitions, never from a vLLM, AITER or Triton kernel.

Rounding points a kernel documents are explicit calls here (`bf16_rne`,
`to_e4m3_rne`, the E8M0 code rules), so a test reproduces them on purpose.
`tests/kernels/models/amd/test_mono_primitives.py` proves each rounding and
code rule bit-exact against the device helpers the kernels use.

This is the formats and rounding half. The layer math and the error metrics
arrive with the model tests that call them."""

import torch

F64 = torch.float64
FLT_MIN = 2.0**-126
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1 = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=F64)


# ---------------------------------------------------------------- formats
def e8m0(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.double() - 127)


def _expand(scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    return scale.repeat_interleave(block[0], -2).repeat_interleave(block[1], -1)


def dequant_fp8(q, scales, block=(32, 32), fnuz=False) -> torch.Tensor:
    """E4M3 [..., N, K] (OCP, or FNUZ: the same bytes at half the value) with
    E8M0 codes [..., N / block[0], K / block[1]] -> float64 [..., N, K]."""
    dt = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    return q.view(torch.uint8).view(dt).double() * _expand(e8m0(scales), block)


# ---------------------------------------------------------------- rounding
def round_float(x, mant: int, emin: int, fmax: float | None = None):
    """`x` rounded to nearest-even onto a binary format with `mant`
    mantissa bits and normal exponents >= `emin` (subnormals below),
    saturating at `fmax` if given, else overflowing to inf at 2^(emax+1)."""
    x = x.double()
    if fmax is not None:
        x = x.clamp(-fmax, fmax)
    e = (torch.frexp(x.abs()).exponent.long() - 1).clamp(min=emin)
    q = torch.exp2(e - mant)
    r = torch.where(torch.isfinite(x) & (x != 0), torch.round(x / q) * q, x)
    if fmax is None:
        r = torch.where(r.abs() >= 2.0**128, r * float("inf"), r)
    return r


def bf16_rne(x):
    return round_float(x, 7, -126)


def f32_rne(x):
    return round_float(x, 23, -126)


def to_e4m3_rne(x, fnuz=False, fmax=None):
    """The E4M3 value of `x` clamped to +-fmax (448 OCP, 224 FNUZ), RNE."""
    if fmax is None:
        fmax = 224.0 if fnuz else 448.0
    return round_float(x, 3, -7 if fnuz else -6, fmax)


def _ceil_log2(v):
    m, e = torch.frexp(v.double())
    return torch.where(v > 0, e.long() - (m == 0.5).long(), -(1 << 20))


def ceil_code(v):
    """The E8M0 round-up rule on an f32 value v >= 0: its exponent field, plus
    one if its mantissa field is not zero (the smallest power of two >= v for
    a normal v; 1 for a subnormal, 0 for zero)."""
    v = v.double()
    return torch.where(v < FLT_MIN, (v > 0).long(), _ceil_log2(v) + 127)


def mx_code_vllm(amax, fmax=448.0):
    """VLLM's `mxfp8_e4m3_quantize`: ceil(log2(RN_f32(max(amax, FLT_MIN)
    / fmax))) + 127, clamped to [0, 254] (0 for an all-zero group)."""
    q = f32_rne(amax.double().clamp(min=FLT_MIN) / fmax)
    return (_ceil_log2(q) + 127).clamp(0, 254)
