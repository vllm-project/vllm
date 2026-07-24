# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Math utility functions for vLLM."""

from typing import overload

# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216


@overload
def cdiv(a: int, b: int) -> int: ...


@overload
def cdiv(a: float, b: int) -> float: ...


def cdiv(a: int | float, b: int) -> int | float:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


@overload
def round_up(x: int, y: int) -> int: ...


@overload
def round_up(x: float, y: int) -> float: ...


def round_up(x: int | float, y: int) -> int | float:
    """Round up x to the nearest multiple of y."""
    return cdiv(x, y) * y


def round_down(x: int, y: int) -> int:
    """Round down x to the nearest multiple of y."""
    return (x // y) * y


def largest_power_of_2_divisor(n: int) -> int:
    """Return the largest power-of-2 that divides *n* (isolate lowest set bit)."""
    return n & (-n)
