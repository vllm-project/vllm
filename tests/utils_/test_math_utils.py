# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.math_utils import (
    cdiv,
    largest_power_of_2_divisor,
    next_power_of_2,
    round_down,
    round_up,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (7, 3, 3),
        (6, 3, 2),
        (0, 3, 0),
        (1, 1, 1),
        (5, 5, 1),
        (10, 3, 4),
        (-7, 3, -2),
        (-6, 3, -2),
        (7, -3, -2),
        (-7, -3, 3),
        (100, 1, 100),
        (1, 100, 1),
    ],
)
def test_cdiv(a: int, b: int, expected: int):
    assert cdiv(a, b) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (7, 8),
        (8, 8),
        (9, 16),
        (15, 16),
        (16, 16),
        (17, 32),
        (1023, 1024),
        (1024, 1024),
        (1025, 2048),
        (2**30, 2**30),
        (2**30 + 1, 2**31),
    ],
)
def test_next_power_of_2(n: int, expected: int):
    assert next_power_of_2(n) == expected


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 4, 0),
        (1, 4, 4),
        (3, 4, 4),
        (4, 4, 4),
        (5, 4, 8),
        (7, 4, 8),
        (8, 4, 8),
        (10, 3, 12),
        (9, 3, 9),
        (100, 10, 100),
        (101, 10, 110),
        (7, 7, 7),
        (8, 7, 14),
    ],
)
def test_round_up(x: int, y: int, expected: int):
    assert round_up(x, y) == expected


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 4, 0),
        (1, 4, 0),
        (3, 4, 0),
        (4, 4, 4),
        (5, 4, 4),
        (7, 4, 4),
        (8, 4, 8),
        (10, 3, 9),
        (9, 3, 9),
        (100, 10, 100),
        (109, 10, 100),
        (7, 7, 7),
        (13, 7, 7),
    ],
)
def test_round_down(x: int, y: int, expected: int):
    assert round_down(x, y) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 1),
        (4, 4),
        (5, 1),
        (6, 2),
        (7, 1),
        (8, 8),
        (12, 4),
        (16, 16),
        (18, 2),
        (20, 4),
        (24, 8),
        (32, 32),
        (48, 16),
        (64, 64),
        (100, 4),
        (128, 128),
        (1024, 1024),
        (0, 0),  # 0 & (-0) == 0
    ],
)
def test_largest_power_of_2_divisor(n: int, expected: int):
    assert largest_power_of_2_divisor(n) == expected
