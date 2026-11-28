# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inputs that make a rounding rule observable, and the comparison of what
comes back. Plain torch, no device code.

A rounding, a packing or a scale code is only wrong somewhere: at a tie, at a
subnormal, at the overflow edge, at the value whose quotient by the format
maximum is a power of two. Random inputs miss all of those, so the generators
here enumerate them, and the random draws are there to catch what the
enumeration forgot rather than the other way round.

The float64 definitions these are checked against are in `reference.py`. The
device side, which is the only part that needs FlyDSL, is in `harness.py`.
"""

import math
from typing import NamedTuple

import torch

from tests.models.common.specialized_decode import reference as R
from vllm.platforms import current_platform

# "cuda" on ROCm too, since torch-hip keeps the name. Asked of the platform
# rather than hardcoded so these run wherever the test does; `or` because an
# unspecified platform's device_type is "", which torch rejects.
DEV = current_platform.device_type or "cpu"
F32, I32 = torch.float32, torch.int32


def bits(x: torch.Tensor) -> torch.Tensor:
    """`x` as float32, as its int32 bit patterns, so that a comparison sees a
    signed zero and distinguishes one ulp rather than printing two equal
    decimals."""
    return x.float().view(I32)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


def _log_uniform(
    n: int, lo: float, hi: float, g: torch.Generator, signed: bool = True
) -> torch.Tensor:
    """`n` float32 with exponents uniform on [lo, hi], so every binade is
    sampled equally rather than the top one taking almost every draw."""
    x = torch.exp2(torch.empty(n, dtype=torch.float64, device=DEV).uniform_(
        lo, hi, generator=g))  # fmt: skip
    if signed:
        x = x * (torch.randint(0, 2, (n,), device=DEV, generator=g) * 2 - 1)
    return x.float()


# Generators take a seed and draw all of a call site's operands in one call,
# so that two operands can't accidentally come from the same seed.
def log_uniform(
    count: int,
    low_exponent: float,
    high_exponent: float,
    seed: int,
    signed: bool = True,
) -> torch.Tensor:
    """`count` float32 with exponents uniform on [low_exponent,
    high_exponent], so every binade is sampled equally."""
    return _log_uniform(count, low_exponent, high_exponent, _gen(seed), signed)


def mixed_floats(
    count: int,
    low: float,
    high: float,
    log_exponents: tuple[float, float],
    seed: int,
) -> torch.Tensor:
    """Half of `count` uniform on [`low`, `high`] (the range a helper is used
    over), half log-uniform over +-2^`log_exponents` (to reach small
    magnitudes)."""
    g = _gen(seed)
    uniform = torch.empty(count // 2, device=DEV).uniform_(low, high, generator=g)
    rest = _log_uniform(count - count // 2, log_exponents[0], log_exponents[1], g)
    return torch.cat([uniform, rest])


def edge_floats() -> torch.Tensor:
    """Every f32 exponent field with the mantissas next to the code / rounding
    boundaries, positive."""
    mant = torch.tensor(
        [0, 1, 2, 0x200000, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFE, 0x7FFFFF]
    )
    bit_patterns = (torch.arange(255)[:, None] << 23) | mant
    return bit_patterns.flatten().to(I32).view(F32).to(DEV)


def neighbours(x: torch.Tensor) -> torch.Tensor:
    """`x` (float32) and its f32 neighbours."""
    inf = torch.full_like(x, math.inf)
    return torch.cat([torch.nextafter(x, -inf), x, torch.nextafter(x, inf)])


def fp8_bytes(fnuz: bool) -> torch.Tensor:
    """Every E4M3 byte that is not a NaN."""
    b = torch.arange(256, device=DEV)
    nan = b == 0x80 if fnuz else (b & 0x7F) == 0x7F
    return b[~nan].to(torch.uint8)


def fp8_domain(fnuz: bool) -> torch.Tensor:
    """Every E4M3 magnitude and every midpoint between neighbours, each with
    its f32 neighbours, both signs; past the range up to inf; 2^20 random."""
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    codes = torch.arange(0, 127, dtype=torch.uint8).view(fp8_dtype)
    magnitudes = codes.double().to(DEV)
    midpoints = (magnitudes[1:] + magnitudes[:-1]) / 2
    past_max = torch.tensor([464.0, 480, 1e6, 3e38], device=DEV)
    values = neighbours(torch.cat([magnitudes, midpoints, past_max]).float())
    random_values = log_uniform(1 << 20, -14, 10, seed=0)
    infinities = torch.tensor([math.inf, -math.inf], device=DEV)
    values = torch.cat([values, -values, random_values, infinities])
    return torch.cat([values, values.new_tensor([1e-40, -1e-40])])


def all_byte_words(exclude: tuple[int, ...] = ()) -> torch.Tensor:
    """Words covering every allowed byte value in every byte position, with
    random other bytes."""
    b = torch.tensor([v for v in range(256) if v not in exclude], device=DEV)
    allowed = b.to(torch.uint8)
    g = _gen(10)
    words = []
    for pos in range(4):
        w = allowed[torch.randint(0, len(b), (len(b), 4), device=DEV, generator=g)]
        w[:, pos] = allowed
        words.append(w)
    return torch.cat(words).contiguous().view(I32).flatten()


def e2m1_bytes(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 values of the 8 nibbles of int32 words, low nibble first."""
    nib = (codes.long()[:, None] >> (4 * torch.arange(8, device=DEV))) & 0xF
    return R.E2M1_VALUES.to(DEV)[nib]


# ------------------------------------------------- MFMA operands and layouts
# Lane layouts of the CDNA MFMA operands, as documented in the ISA.

# Operand format codes of the f8f6f4 MFMA (CDNA4 ISA).
MX_FP8, MX_FP4 = 0, 4


def mfma_lanes(tile: torch.Tensor, per_lane: int) -> torch.Tensor:
    """[tiles, 16, K] -> the [tiles 64, per_lane] lane operands of a 16 x 16
    MFMA: lane l holds row l % 16, K per_lane (l / 16) .. + per_lane."""
    tiles = tile.shape[0]
    return (
        tile.view(tiles, 16, 4, per_lane)
        .permute(0, 2, 1, 3)
        .reshape(tiles * 64, per_lane)
    )


def acc_lanes(acc: torch.Tensor) -> torch.Tensor:
    """[tiles, 16, 16] accumulator -> the [tiles 64, 4] lane operands: lane l
    holds rows 4 (l / 16) .. + 3 of column l % 16."""
    return acc.view(-1, 4, 4, 16).transpose(2, 3).reshape(-1, 4)


def from_acc_lanes(lanes: torch.Tensor) -> torch.Tensor:
    """`acc_lanes` undone: [tiles 64, 4] -> [tiles, 16, 16]."""
    return lanes.view(-1, 4, 16, 4).transpose(2, 3).reshape(-1, 16, 16)


def _fp8_mfma_lanes(quantized: torch.Tensor) -> torch.Tensor:
    """[tiles, 16, 128] E4M3 -> the f8f6f4 MFMA's lane operands: lane l holds
    row l % 16, K 16 j .. + 16 and 64 + 16 j .. + 16, for j = l / 16."""
    tiles = quantized.shape[0]
    halves = quantized.view(tiles, 16, 2, 4, 16).permute(0, 3, 1, 2, 4)
    return halves.reshape(tiles * 64, 32).contiguous()


class Bf16MfmaOperands(NamedTuple):
    """Operands of a 16 x 16 x 32 bf16 MFMA, as matrices rather than lanes."""

    a: torch.Tensor  # [tiles, 16, 32] bfloat16
    b: torch.Tensor  # [tiles, 16, 32] bfloat16
    acc: torch.Tensor  # [tiles, 16, 16] float32


def _wide_bf16(
    shape: tuple[int, ...], exponents: tuple[int, int], g: torch.Generator
) -> torch.Tensor:
    x = torch.randn(shape, device=DEV, generator=g)
    spread = torch.exp2(torch.randint(*exponents, shape, device=DEV, generator=g))
    return (x * spread).bfloat16()


def wide_bf16(
    shape: tuple[int, ...], exponents: tuple[int, int], seed: int
) -> torch.Tensor:
    """bf16 normals of `shape` scaled by random powers of two in `exponents`,
    so that a float32 accumulation of them actually rounds."""
    return _wide_bf16(shape, exponents, _gen(seed))


def bf16_mfma_operands(tiles: int, seed: int) -> Bf16MfmaOperands:
    """A, B and C for a 16 x 16 x 32 bf16 MFMA."""
    g = _gen(seed)
    a, b = (_wide_bf16((tiles, 16, 32), (-20, 20), g) for _ in range(2))
    return Bf16MfmaOperands(a, b, torch.randn(tiles, 16, 16, device=DEV, generator=g))


class MxMfmaOperands(NamedTuple):
    """Operands of a 16 x 16 x 128 scaled (f8f6f4) MFMA. The `*_values` are
    the exact unscaled operands in float64."""

    a_words: torch.Tensor  # [tiles 64, n] int32 lane operands
    b_words: torch.Tensor
    a_values: torch.Tensor  # [tiles, 16, 128] float64
    b_values: torch.Tensor
    scale_words: torch.Tensor  # [2, tiles 64] int32, four E8M0 codes a word
    scale_codes: torch.Tensor  # [2, tiles 64, 4], the codes those words hold
    acc: torch.Tensor  # [tiles, 16, 16] float32


def _mx_operand(
    tiles: int, fmt: int, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random MXFP8 (no NaN) or MXFP4 `[tiles, 16, 128]` -> (lane words,
    unscaled float64 values)."""
    assert fmt in (MX_FP8, MX_FP4), f"unknown f8f6f4 operand format {fmt}"
    if fmt == MX_FP8:
        quantized = torch.randint(0, 256, (tiles, 16, 128), device=DEV, generator=g)
        quantized = torch.where((quantized & 0x7F) == 0x7F, 0, quantized)
        quantized = quantized.to(torch.uint8)
        values = quantized.view(torch.float8_e4m3fn).double()
        return _fp8_mfma_lanes(quantized).view(I32), values
    packed = torch.randint(
        0, 256, (tiles, 16, 64), device=DEV, generator=g, dtype=torch.uint8
    )
    # E8M0 code 127 is a scale of 1.
    unscaled = packed.new_full((tiles, 16, 4), 127)
    words = mfma_lanes(packed, 16).contiguous().view(I32)
    return words, R.dequant_mxfp4(packed, unscaled)


def mx_mfma_operands(tiles: int, formats: tuple[int, int], seed: int) -> MxMfmaOperands:
    """A, B, their E8M0 scale words and C for a 16 x 16 x 128 scaled MFMA.

    Scale codes span 100 .. 154 so that selecting the wrong byte of a scale
    word changes the result.
    """
    g = _gen(seed)
    (a_words, a_values), (b_words, b_values) = (
        _mx_operand(tiles, fmt, g) for fmt in formats
    )
    codes = torch.randint(100, 155, (2, tiles * 64, 4), device=DEV, generator=g)
    # Reinterpret the four bytes as one int32 word.
    words = codes.to(torch.uint8).contiguous().view(I32).squeeze(-1)
    return MxMfmaOperands(
        a_words,
        b_words,
        a_values,
        b_values,
        words,
        codes,
        torch.randn(tiles, 16, 16, device=DEV, generator=g),
    )


def assert_bit_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    helper: str,
    sent: torch.Tensor,
) -> None:
    """`actual == expected` row for row, naming the helper and the first rows
    that differ.

    `sent` is what was fed to the helper, one row per row of `actual`. Plain
    `torch.equal` is enough where the input is a handful of values and is used
    directly there; the generators above run to a million rows, and a bare
    `False` over a million roundings says nothing about which one broke.
    """
    # Not just a sanity check: `!=` would broadcast a mismatch into a square
    # of comparisons and the report below would then be about rows that were
    # never computed.
    assert actual.shape == expected.shape, (
        f"{helper}: got shape {tuple(actual.shape)}, want {tuple(expected.shape)}"
    )
    differs = actual != expected
    if actual.is_floating_point():
        # -0.0 == +0.0, so a dropped zero sign would pass on `!=` alone, and
        # OCP's 0x80 is a negative zero that `fp8_bytes` feeds in.
        differs |= torch.signbit(actual) != torch.signbit(expected)
    bad = differs.view(len(expected), -1).any(1).nonzero().flatten()[:3]
    assert bad.numel() == 0, (
        f"{helper} differs at {sent[bad].tolist()}: "
        f"got {actual[bad].tolist()}, want {expected[bad].tolist()}"
    )
