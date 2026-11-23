# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E4M3 conversions of the Triton kernels. They use Triton's fp8e4nv
conversions on SM90+ only (`use_native_e4m3`): below, launchers pass e4m3
tensors as uint8 (`e4m3_arg`) and the kernels convert in software, as the
SM90 hardware rounds."""

import torch

from vllm.triton_utils import tl, tldevice, triton

from . import use_native_e4m3

_NATIVE_E4M3 = tl.constexpr(use_native_e4m3())


def e4m3_arg(x: torch.Tensor | None) -> torch.Tensor | None:
    """`x` as a kernel argument: e4m3 tensors are uint8 below SM90."""
    if x is not None and x.dtype == torch.float8_e4m3fn and not use_native_e4m3():
        return x.view(torch.uint8)
    return x


@triton.jit
def _e4m3_bits(x):
    """The e4m3 byte of fp32 `x`, rounded to nearest even with saturation
    (`cvt.rn.satfinite`)."""
    sign = (x.to(tl.uint32, bitcast=True) >> 24) & 0x80
    a = tl.minimum(tl.abs(x), 448.0)
    bits = a.to(tl.uint32, bitcast=True)
    # Round the fp32 mantissa to 3 bits and rebias the exponent (127 -> 7).
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - (120 << 3)
    # Subnormals are multiples of 2**-9; 8 of them encode 2**-6.
    subnormal = tldevice.rint(a * 512.0).to(tl.uint32)
    code = tl.where(a < 0.015625, subnormal, normal)
    code = tl.where(x != x, 0x7F, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _e4m3_value(b):
    """The fp32 value of e4m3 byte `b`."""
    b = b.to(tl.uint32)
    magnitude = b & 0x7F
    normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    value = tl.where(magnitude < 8, magnitude.to(tl.float32) * (1.0 / 512.0), normal)
    value = tl.where(magnitude == 0x7F, float("nan"), value)
    return tl.where((b & 0x80) != 0, -value, value)


@triton.jit
def _e4m3_f16(b):
    """The fp16 bits (uint16) of the e4m3 bytes in uint16 `b`, as
    `cvt.rn.f16x2.e4m3x2`: 16-bit throughout, for the MMA operand order of
    16-bit loads."""
    b = b.to(tl.uint16)
    magnitude = b & 0x7F
    m = magnitude & 7
    # Subnormal m * 2**-9 is normal in fp16.
    subnormal = tl.where(
        m == 1,
        0x1800,
        tl.where(
            m == 2,
            0x1C00,
            tl.where(
                m == 3,
                0x1E00,
                tl.where(m < 6, 0x2000 + (m - 4) * 0x100, 0x2200 + (m - 6) * 0x100),
            ),
        ),
    ).to(tl.uint16)
    subnormal = tl.where(m == 0, 0, subnormal).to(tl.uint16)
    normal = ((magnitude << 7) + (8 << 10)).to(tl.uint16)
    bits = tl.where(magnitude < 8, subnormal, normal)
    bits = tl.where(magnitude == 0x7F, 0x7E00, bits).to(tl.uint16)
    return bits | ((b & 0x80) << 8).to(tl.uint16)


@triton.jit
def to_e4m3(x):
    """`x` as e4m3, uint8 bits below SM90."""
    # One expression: Triton also types code after a constexpr branch's return.
    return x.to(tl.float8e4nv) if _NATIVE_E4M3 else _e4m3_bits(x.to(tl.float32))


@triton.jit
def from_e4m3(x):
    """fp32 values of e4m3 `x` (uint8 bits below SM90)."""
    return x.to(tl.float32) if _NATIVE_E4M3 else _e4m3_value(x)


@triton.jit
def cast_store(x, dtype: tl.constexpr):
    """`x` cast to pointer element type `dtype`, where uint8 stands for e4m3
    below SM90."""
    return _e4m3_bits(x.to(tl.float32)) if dtype == tl.uint8 else x.to(dtype)
