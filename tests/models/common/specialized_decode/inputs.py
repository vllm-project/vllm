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
    random_values = _log_uniform(1 << 20, -14, 10, _gen(0))
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
