# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helpers for OCP MXFP4 KV-cache stores: group-32 E2M1 elements with
one E8M0 scale per group.

The scale is ``ceil_pow2(amax / 6)``, so a group's peak lands in (3, 6] and
never saturates, and the element rounding is round-half-to-even, matching
AITER's reference quantizer bit for bit.
"""

from vllm.triton_utils import tl, triton

E8M0_BIAS = tl.constexpr(127)

# E2M1 magnitudes by code index 0..7. Code is ``sign << 3 | index``.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = tl.constexpr(_E2M1_VALUES[-1])

# Midpoints between consecutive E2M1 magnitudes. Counting the midpoints below
# ``a`` rounds to nearest; a value on a midpoint goes to the even code index,
# so ``_M1``, ``_M3``, ``_M5`` compare with ``>=``.
_M0, _M1, _M2, _M3, _M4, _M5, _M6 = (
    tl.constexpr((lo + hi) / 2) for lo, hi in zip(_E2M1_VALUES, _E2M1_VALUES[1:])
)


@triton.jit
def ceil_pow2_exponent(v):
    """Smallest integer ``e`` with ``2**e >= v``, for positive fp32 ``v``.

    Done on the bit pattern rather than with ``log2``: fp32 ``log2`` is not
    exact at powers of two, and rounding one of those up would waste a whole
    exponent step and halve the usable E2M1 range for that group.
    """
    bits = v.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x7FFFFF
    return tl.where(mantissa == 0, exponent, exponent + 1)


@triton.jit
def quantize_e2m1(normalized):
    """fp32 values pre-divided by their group scale -> 4-bit E2M1 codes."""
    a = tl.abs(normalized)
    index = (
        (a > _M0).to(tl.int32)
        + (a >= _M1).to(tl.int32)
        + (a > _M2).to(tl.int32)
        + (a >= _M3).to(tl.int32)
        + (a > _M4).to(tl.int32)
        + (a >= _M5).to(tl.int32)
        + (a > _M6).to(tl.int32)
    )
    # The sign bit is kept when the magnitude rounds to zero, so a small
    # negative encodes as -0.0 (code 8), as AITER's reference does.
    sign = (normalized < 0).to(tl.int32) << 3
    return (index | sign).to(tl.uint8)


@triton.jit
def mxfp4_group_scale_code(absmax):
    """Per-group E8M0 scale code for the groups' ``absmax``.

    Clamped to [1, 254] so no scale decodes to zero or NaN; an all-zero group
    is pinned to 2**0 so its encoding is deterministic.
    """
    shared_exp = ceil_pow2_exponent(absmax / E2M1_MAX)
    encoded = tl.minimum(tl.maximum(shared_exp + E8M0_BIAS, 1), 254)
    return tl.where(absmax == 0, E8M0_BIAS, encoded)
