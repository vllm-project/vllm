# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `vllm/v1/attention/ops/kitty/tiles.py`."""

import numpy as np
import pytest

from vllm.v1.attention.ops.kitty.tiles import (
    KITTY_PAGE_SIZE,
    kitty_d_boost,
    kitty_first_unsealed_k,
    kitty_first_unsealed_v,
    kitty_page_bytes,
    kitty_state_content_bytes,
)

HEAD_SIZES = [64, 80, 96, 128, 256]
BOOST_RATIOS = [0.0, 0.1, 0.25, 0.5, 1.0]


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


@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("boost_ratio", BOOST_RATIOS)
def test_d_boost_stays_within_head_size(head_size, boost_ratio):
    """Ratios in the allowed [0, 1] range never boost more channels than exist."""
    assert 0 <= kitty_d_boost(head_size, boost_ratio) <= head_size


# kitty_page_bytes


def page_bytes_from_layout(head_size: int, d_boost: int) -> int:
    """Sum the page segments from their shapes and element widths."""
    tokens = KITTY_PAGE_SIZE
    two_bits, fp16, uint8 = 2 / 8, 2, 1
    segments = [
        head_size * tokens * two_bits,  # K low plane
        tokens * head_size * two_bits,  # V data
        d_boost * tokens * two_bits,  # K high plane
        head_size * 2 * fp16,  # K scale/min
        tokens * 2 * fp16,  # V scale/min
        head_size * uint8,  # K rank table
    ]
    return int(sum(segments))


def test_page_bytes_default_config():
    """D = 128 with the default ratio 0.25 (DB = 32)."""
    assert kitty_page_bytes(128, 32) == 10368


@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("boost_ratio", BOOST_RATIOS)
def test_page_bytes_matches_segment_layout(head_size, boost_ratio):
    d_boost = kitty_d_boost(head_size, boost_ratio)
    assert kitty_page_bytes(head_size, d_boost) == page_bytes_from_layout(
        head_size, d_boost
    )


@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_page_bytes_each_boosted_channel_costs_32_bytes(head_size):
    """A boosted channel adds one 128-token row of 2-bit codes to the high plane."""
    for d_boost in range(head_size):
        growth = kitty_page_bytes(head_size, d_boost + 1) - kitty_page_bytes(
            head_size, d_boost
        )
        assert growth == 32


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


@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("boost_ratio", BOOST_RATIOS)
def test_state_content_bytes_is_smallest_slot_size_that_fits_page(
    head_size, boost_ratio
):
    """128 slots of C bytes must hold the page, with less than one slot of padding."""
    d_boost = kitty_d_boost(head_size, boost_ratio)
    page_bytes = kitty_page_bytes(head_size, d_boost)
    content = kitty_state_content_bytes(head_size, d_boost)
    assert KITTY_PAGE_SIZE * content >= page_bytes
    assert KITTY_PAGE_SIZE * (content - 1) < page_bytes


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


def test_first_unsealed_k_tail_fits_in_one_page():
    """Unsealed K positions past the sink fit the 128-row K tail buffer."""
    n = np.arange(KITTY_PAGE_SIZE, 10 * KITTY_PAGE_SIZE)
    tail_tokens = n - KITTY_PAGE_SIZE * kitty_first_unsealed_k(n)
    np.testing.assert_array_equal(tail_tokens, n % KITTY_PAGE_SIZE)


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


def test_first_unsealed_v_tail_fits_in_two_pages():
    """Unsealed V positions past the sink fit the 256-row V tail buffer."""
    n = np.arange(2 * KITTY_PAGE_SIZE, 10 * KITTY_PAGE_SIZE)
    tail_tokens = n - KITTY_PAGE_SIZE * kitty_first_unsealed_v(n)
    np.testing.assert_array_equal(tail_tokens, KITTY_PAGE_SIZE + n % KITTY_PAGE_SIZE)
