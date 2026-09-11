# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Math utility functions for vLLM."""

from dataclasses import dataclass

# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    """Round down x to the nearest multiple of y."""
    return (x // y) * y


def largest_power_of_2_divisor(n: int) -> int:
    """Return the largest power-of-2 that divides *n* (isolate lowest set bit)."""
    return n & (-n)


@dataclass(slots=True)
class WelfordAccumulator:
    """Single-pass count, exact integer sum, and population variance."""

    count: int = 0
    total: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        delta = value - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (value - self._mean)

    @property
    def variance(self) -> float:
        return self._m2 / self.count if self.count else 0.0
