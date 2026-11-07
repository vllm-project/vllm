# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.math_utils import (
    largest_power_of_2_divisor,
    prev_power_of_2,
    round_up_to,
)


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (-3, 0),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (7, 4),
        (8, 8),
        (9, 8),
        (16, 16),
        (100, 64),
        (1024, 1024),
        (1025, 1024),
    ],
)
def test_prev_power_of_2(n, expected):
    assert prev_power_of_2(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 1),
        (4, 4),
        (6, 2),
        (7, 1),
        (12, 4),
        (16, 16),
        (24, 8),
    ],
)
def test_largest_power_of_2_divisor(n, expected):
    assert largest_power_of_2_divisor(n) == expected


@pytest.mark.parametrize(
    "values,n,expected",
    [
        ([2, 4, 8], 1, 2),
        ([2, 4, 8], 2, 2),
        ([2, 4, 8], 3, 4),
        ([2, 4, 8], 8, 8),
        ([2, 4, 8], 9, None),
        ([], 5, None),
        ([5], 5, 5),
        ([5], 6, None),
        ([1, 3, 6, 10], 7, 10),
    ],
)
def test_round_up_to(values, n, expected):
    assert round_up_to(values, n) == expected
