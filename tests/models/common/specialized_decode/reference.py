# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the formats a specialized decode uses are defined to do, in float64.

Any model's decode tests can use these: a test names the rule it expects
(`bf16_rne`, `to_e4m3_rne`, an E8M0 code rule) and compares its kernel against
it, rather than inheriting whatever rounding some other implementation
happens to do. `tests/models/deepseek_v41/amd/mono/test_primitives.py` is
today's only caller, and holds every rule below bit-exact.

A definition must not be derived from the implementation it checks, nor reach
for a helper that implementation also reaches for, or a shared bug cancels and
the test passes. That is about where a definition comes from, not about which
library writes it down.

Formats are all that the primitive tier (T1) of #68016 needs. The layer math
and the error metrics land here next, shared rather than once per model, so
that the `DecodeNumerics` contract (T2) measures every model the same way."""

import torch

F64 = torch.float64
# The smallest positive normal float32, which the E8M0 code rules floor at,
# under the name the device code gives it (attention/device.py, stages/moe.py).
FLT_MIN = 2.0**-126
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Indexed by the 4-bit E2M1 code: the sign is bit 3, so the negatives follow
# the positives in the same order.
E2M1_VALUES = torch.tensor(
    _E2M1_MAGNITUDES + tuple(-v for v in _E2M1_MAGNITUDES), dtype=F64
)
_FP8_BYTE_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
)


# ---------------------------------------------------------------- formats
def e8m0(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.double() - 127)


def _expand(scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    return scale.repeat_interleave(block[0], -2).repeat_interleave(block[1], -1)


def dequant_fp8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block: tuple[int, int],
    fnuz: bool,
) -> torch.Tensor:
    """E4M3 [..., N, K] (OCP, or FNUZ: the same bytes at half the value) with
    E8M0 codes [..., N / block[0], K / block[1]] -> float64 [..., N, K]."""
    assert quantized.dtype in _FP8_BYTE_DTYPES, (
        f"`quantized` holds fp8 bytes, got {quantized.dtype}"
    )
    assert not scales.is_floating_point(), (
        f"`scales` holds E8M0 codes, got {scales.dtype}"
    )
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    values = quantized.view(torch.uint8).view(fp8_dtype).double()
    return values * _expand(e8m0(scales), block)


# ---------------------------------------------------------------- rounding
def round_float(
    x: torch.Tensor,
    mantissa_bits: int,
    min_exponent: int,
    fmax: float | None = None,
) -> torch.Tensor:
    """`x` rounded to nearest-even onto a binary format with `mantissa_bits`
    mantissa bits and normal exponents >= `min_exponent` (subnormals below),
    saturating at `fmax` if given, else overflowing to inf at 2^(emax+1)."""
    x = x.double()
    if fmax is not None:
        x = x.clamp(-fmax, fmax)
    exponent = (torch.frexp(x.abs()).exponent.long() - 1).clamp(min=min_exponent)
    # float64, or exp2 of an integer tensor gives the quantum in the default
    # dtype and 2^(exponent - mantissa_bits) is past float32's range for a
    # large x.
    quantum = torch.exp2((exponent - mantissa_bits).double())
    rounded = torch.where(
        torch.isfinite(x) & (x != 0), torch.round(x / quantum) * quantum, x
    )
    if fmax is None:
        rounded = torch.where(
            rounded.abs() >= 2.0**128, rounded * float("inf"), rounded
        )
    return rounded


def bf16_rne(x: torch.Tensor) -> torch.Tensor:
    return round_float(x, 7, -126)


def f32_rne(x: torch.Tensor) -> torch.Tensor:
    return round_float(x, 23, -126)


def to_e4m3_rne(x: torch.Tensor, fnuz: bool) -> torch.Tensor:
    """The E4M3 value of `x` clamped to the format maximum (448 OCP, 224
    FNUZ), RNE."""
    return round_float(x, 3, -7 if fnuz else -6, 224.0 if fnuz else 448.0)


def _ceil_log2(value: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(value.double())
    return torch.where(
        value > 0, exponent.long() - (mantissa == 0.5).long(), -(1 << 20)
    )


def ceil_code(value: torch.Tensor) -> torch.Tensor:
    """The E8M0 round-up rule on an f32 `value` >= 0: its exponent field, plus
    one if its mantissa field is not zero (the smallest power of two >= the
    value for a normal one; 1 for a subnormal, 0 for zero)."""
    value = value.double()
    return torch.where(value < FLT_MIN, (value > 0).long(), _ceil_log2(value) + 127)


def mx_code_vllm(amax: torch.Tensor, fmax: float) -> torch.Tensor:
    """VLLM's `mxfp8_e4m3_quantize`: ceil(log2(RN_f32(max(amax, FLT_MIN)
    / fmax))) + 127, clamped to [0, 254] (0 for an all-zero group)."""
    scaled = f32_rne(amax.double().clamp(min=FLT_MIN) / fmax)
    return (_ceil_log2(scaled) + 127).clamp(0, 254)
