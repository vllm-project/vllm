# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `vllm/v1/attention/ops/kitty/tiles.py`."""

import numpy as np
import pytest

from vllm.v1.attention.ops.kitty.tiles import (
    K_TAIL_ROW,
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_ROWS,
    V_SINK_ROW,
    V_TAIL_ROW,
    kitty_d_boost,
    kitty_first_unsealed_k,
    kitty_first_unsealed_v,
    kitty_page_bytes,
    kitty_state_content_bytes,
)

# kitty_d_boost


@pytest.mark.parametrize(
    "head_size, boost_ratio, expected",
    [
        (128, 0.25, 32),  # the default kitty_boost_ratio
        (128, 0.0, 0),  # no boosted channels
        (128, 1.0, 128),  # every channel boosted
        (64, 0.3, 19),  # 19.2 rounds down
        (128, 0.199, 25),  # 25.47 rounds down, not to nearest
    ],
)
def test_d_boost_is_floor_of_head_size_times_ratio(head_size, boost_ratio, expected):
    d_boost = kitty_d_boost(head_size, boost_ratio)
    assert d_boost == expected
    assert isinstance(d_boost, int)


# kitty_page_bytes


@pytest.mark.parametrize(
    "head_size, d_boost, expected",
    [
        (128, 32, 10368),  # the default kitty_boost_ratio
        (128, 0, 9344),  # no K high plane
        (128, 128, 13440),  # every K channel in the high plane
        (64, 16, 5440),
        (80, 20, 6672),
        (256, 64, 20224),
    ],
)
def test_page_bytes_known_values(head_size, d_boost, expected):
    assert kitty_page_bytes(head_size, d_boost) == expected


# kitty_state_content_bytes


@pytest.mark.parametrize(
    "head_size, d_boost, expected",
    [
        (128, 32, 81),  # 10368 bytes is exactly 81 token slots
        (64, 16, 43),  # 5440 bytes is 42.5 slots, padded to 43
        (80, 20, 53),  # 6672 bytes is 52.125 slots, padded to 53
    ],
)
def test_state_content_bytes_known_values(head_size, d_boost, expected):
    assert kitty_state_content_bytes(head_size, d_boost) == expected


# kitty_first_unsealed_k

# (cache length n, first unsealed K block)
FIRST_UNSEALED_K = [
    (0, 1),
    (1, 1),
    (127, 1),  # sink block not full yet
    (128, 1),  # sink full, but the sink is never sealed
    (255, 1),  # block 1 one token short of full
    (256, 2),  # block 1 full, so sealed
    (257, 2),
    (383, 2),
    (384, 3),
    (1000, 7),
]


@pytest.mark.parametrize("n, expected", FIRST_UNSEALED_K)
def test_first_unsealed_k_scalar(n, expected):
    assert kitty_first_unsealed_k(n) == expected


def test_first_unsealed_k_array_is_elementwise():
    """`plan_kitty_step` passes per-request cache lengths as one array."""
    lengths, expected = zip(*FIRST_UNSEALED_K)
    got = kitty_first_unsealed_k(np.array(lengths, dtype=np.int64))
    np.testing.assert_array_equal(got, expected)


def test_first_unsealed_k_tail_fits_k_tail_buffer():
    """Unsealed K positions past the sink never outnumber the K tail rows."""
    n = np.arange(KITTY_PAGE_SIZE, 10 * KITTY_PAGE_SIZE)
    tail_tokens = n - KITTY_PAGE_SIZE * kitty_first_unsealed_k(n)
    assert tail_tokens.min() == 0
    assert tail_tokens.max() == V_SINK_ROW - K_TAIL_ROW - 1


# kitty_first_unsealed_v

# (cache length n, first unsealed V block)
FIRST_UNSEALED_V = [
    (0, 1),
    (1, 1),
    (127, 1),  # sink block not full yet
    (128, 1),  # sink full, but the sink is never sealed
    (256, 1),  # block 1 full, but stays unsealed as the latest full block
    (383, 1),  # block 2 one token short of full
    (384, 2),  # block 2 full, so block 1 is sealed
    (385, 2),
    (511, 2),
    (512, 3),
    (1000, 6),
]


@pytest.mark.parametrize("n, expected", FIRST_UNSEALED_V)
def test_first_unsealed_v_scalar(n, expected):
    assert kitty_first_unsealed_v(n) == expected


def test_first_unsealed_v_array_is_elementwise():
    """`plan_kitty_step` passes per-request cache lengths as one array."""
    lengths, expected = zip(*FIRST_UNSEALED_V)
    got = kitty_first_unsealed_v(np.array(lengths, dtype=np.int64))
    np.testing.assert_array_equal(got, expected)


def test_first_unsealed_v_tail_fits_v_tail_buffer():
    """Past the first two blocks, the V tail keeps at least one full unsealed
    block and never outnumbers the V tail rows."""
    n = np.arange(2 * KITTY_PAGE_SIZE, 10 * KITTY_PAGE_SIZE)
    tail_tokens = n - KITTY_PAGE_SIZE * kitty_first_unsealed_v(n)
    assert tail_tokens.min() == KITTY_PAGE_SIZE
    assert tail_tokens.max() == KITTY_RESIDUAL_ROWS - V_TAIL_ROW - 1
