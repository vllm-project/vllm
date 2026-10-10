# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 codepoint + UE8M0 scale constants for the UltraQuant KV cache.

K/V codes are FP4 E2M1. Per-group scales are UE8M0 (one byte, a power of
two): ``s = 2^round(log2(c · absmax))`` with ``c = 0.156``. Decode
consumes Q as FP8 E4M3 so scaled F8F6F4 MFMA can run natively on CDNA4.

AoS slot layout (group_size=32), per (token, head):
    bytes [0 .. D/2)            : K codes (FP4 nibbles, 2/byte)
    bytes [D/2 .. D/2 + Gk)     : K scales (UE8M0, 1 byte × Gk groups)
    bytes [D/2 + Gk .. D + Gk)  : V codes
    bytes [D + Gk .. D + 2*Gk)  : V scales
    D=256 → Gk=8 → 272 B/slot. D=128 → Gk=4 → 136 B/slot.

No per-token norm-fold. No V rotation. K is Hadamard-rotated at store.
"""

from __future__ import annotations

import math

# ── 15 unique FP4 E2M1 levels, sorted ascending ────
FP4_LEVELS_SORTED: tuple[float, ...] = (
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)
assert len(FP4_LEVELS_SORTED) == 15

# Midpoints between consecutive sorted levels (14 boundaries).
MIDPOINTS_SORTED: tuple[float, ...] = tuple(
    (FP4_LEVELS_SORTED[i] + FP4_LEVELS_SORTED[i + 1]) / 2.0
    for i in range(len(FP4_LEVELS_SORTED) - 1)
)
assert len(MIDPOINTS_SORTED) == 14

# FP4 E2M1 bit-pattern decode (16 entries indexed by 4-bit code).
# Sign bit (MSB) | 2 exp bits | 1 mantissa bit (LSB).
FP4_BITS_TO_VALUE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
assert len(FP4_BITS_TO_VALUE) == 16

# Sorted index → FP4 E2M1 bit pattern (uint4).
SORTED_TO_BITS: tuple[int, ...] = (
    0b1111,
    0b1110,
    0b1101,
    0b1100,
    0b1011,
    0b1010,
    0b1001,
    0b0000,
    0b0001,
    0b0010,
    0b0011,
    0b0100,
    0b0101,
    0b0110,
    0b0111,
)
assert len(SORTED_TO_BITS) == 15
for _i, _bits in enumerate(SORTED_TO_BITS):
    _v = FP4_BITS_TO_VALUE[_bits]
    _v = 0.0 if _v == 0.0 else _v
    assert _v == FP4_LEVELS_SORTED[_i]

# ── Format constants ──────────────────────────────────────────────────────
GROUP_SIZE: int = 32
"""Elements per group; one E8M0 scale per group. Fixed at 32 because the
AMD scaled F8F6F4 MFMA instruction also consumes one E8M0 scale per 32
elements along K. Other group sizes would force a fallback to plain MFMA
+ accumulator-side scale multiply (no hardware fast-path)."""

FP4_MAX: float = 6.0
"""Largest representable FP4 magnitude."""

DEFAULT_CONSTANT_C: float = 0.156
"""MSE-optimal scale constant; ``s_raw = c * absmax``, then UE8M0-snapped."""


def get_constant_c() -> float:
    """Return the fixed UltraQuant scale constant."""
    return DEFAULT_CONSTANT_C


def get_group_size() -> int:
    """Return the fixed group size required by scaled MFMA."""
    return GROUP_SIZE


# ── Scale encoding ────────────────────────────────────────────────────────
# Per-group scale `s = pow2_round(c · absmax)` stored as the E8M0 byte;
# codebook is the raw FP4 grid {0, ±0.5, ..., ±6}; dequant = code · s.


# ── UE8M0 helpers ──────────────────────────────────────────────────────────
# E8M0 = 8 exponent bits, 0 mantissa bits, no sign. The byte value `e`
# represents `2^(e - 127)` for e in [1, 254]. e=0 and e=255 are reserved
# (zero and NaN respectively in the OCP spec). We use e=0 as our zero
# sentinel for sink-zero groups.

UE8M0_BIAS: int = 127

# fp32 minimum positive normal = 2^-126 ≈ 1.175e-38. Anything smaller
# clamps to the zero sentinel.
_UE8M0_MIN_EXP: int = -126
_UE8M0_MAX_EXP: int = 127


def ue8m0_encode(s: float) -> int:
    """Snap a positive fp32 scale `s` to the nearest power of 2 and
    encode as a UE8M0 byte. `s <= 0` encodes as 0 (zero sentinel)."""
    if s <= 0.0 or not math.isfinite(s):
        return 0
    exp = int(round(math.log2(s)))
    if exp < _UE8M0_MIN_EXP:
        return 0
    if exp > _UE8M0_MAX_EXP:
        exp = _UE8M0_MAX_EXP
    return exp + UE8M0_BIAS


def ue8m0_decode(byte: int) -> float:
    """Decode a UE8M0 byte back to fp32. 0 → 0.0 (zero sentinel)."""
    if byte == 0:
        return 0.0
    return 2.0 ** (byte - UE8M0_BIAS)


# ── Per-slot AoS byte layout ──────────────────────────────────────────────
# Per slot, per head (group_size=GS=32, Gk = D/GS):
#   bytes [0 .. D/2)               : K codes (2 nibbles/byte)
#   bytes [D/2 .. D/2 + Gk)        : K scales (E8M0, 1 byte each)
#   bytes [D/2 + Gk .. D + Gk)     : V codes
#   bytes [D + Gk .. D + 2*Gk)     : V scales (E8M0, 1 byte each)
# D=256 → 272 B/slot. D=128 → 136 B/slot.


def k_codes_bytes(head_dim: int) -> int:
    """Bytes for one head's packed K codes."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    return head_dim // 2


def n_groups(head_dim: int, group_size: int | None = None) -> int:
    gs = group_size if group_size is not None else get_group_size()
    if head_dim % gs != 0:
        raise ValueError(f"head_dim={head_dim} must be a multiple of group_size={gs}")
    return head_dim // gs


def k_scales_bytes(head_dim: int, group_size: int | None = None) -> int:
    """Bytes for one head's K scales (one E8M0 byte per group)."""
    return n_groups(head_dim, group_size)


def v_codes_bytes(head_dim: int) -> int:
    return k_codes_bytes(head_dim)


def v_scales_bytes(head_dim: int, group_size: int | None = None) -> int:
    return k_scales_bytes(head_dim, group_size)


def slot_size(head_dim: int, group_size: int | None = None) -> int:
    """Bytes per (token, head) slot."""
    return (
        k_codes_bytes(head_dim)
        + k_scales_bytes(head_dim, group_size)
        + v_codes_bytes(head_dim)
        + v_scales_bytes(head_dim, group_size)
    )


def k_codes_offset(head_dim: int, group_size: int | None = None) -> int:
    return 0


def k_scales_offset(head_dim: int, group_size: int | None = None) -> int:
    return k_codes_bytes(head_dim)


def v_codes_offset(head_dim: int, group_size: int | None = None) -> int:
    return k_codes_bytes(head_dim) + k_scales_bytes(head_dim, group_size)


def v_scales_offset(head_dim: int, group_size: int | None = None) -> int:
    return v_codes_offset(head_dim, group_size) + v_codes_bytes(head_dim)
