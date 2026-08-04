# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``batched_iteration_with_skip``."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    batched_iteration_with_skip,
)


def test_basic_batching_with_skip():
    """Skipped items are dropped and reported indices stay in original space."""
    data = list(range(10))
    result = list(batched_iteration_with_skip(data, batch_size=3, skip_count=2))

    assert result == [
        (2, (2, 3, 4)),
        (5, (5, 6, 7)),
        (8, (8, 9)),
    ]


def test_skip_count_zero_matches_plain_batching():
    """With skip_count=0 every item is yielded, indexed from 0."""
    data = list(range(7))
    result = list(batched_iteration_with_skip(data, batch_size=2, skip_count=0))

    assert result == [
        (0, (0, 1)),
        (2, (2, 3)),
        (4, (4, 5)),
        (6, (6,)),
    ]
    # The concatenation of all batches equals the unskipped tail of the list.
    flattened = [item for _, batch in result for item in batch]
    assert flattened == data


def test_batch_start_indices_are_original_indices():
    """Reported start index is the original list index, accounting for skip."""
    data = list(range(20))
    result = list(batched_iteration_with_skip(data, batch_size=5, skip_count=10))

    start_indices = [start for start, _ in result]
    assert start_indices == [10, 15]
    # The docstring example: skip_count=10, batch_size=5 -> first start idx 10.
    assert result[0] == (10, (10, 11, 12, 13, 14))


def test_partial_final_batch():
    """The final short batch still reports the correct start index."""
    data = list(range(8))
    result = list(batched_iteration_with_skip(data, batch_size=3, skip_count=1))

    assert result == [
        (1, (1, 2, 3)),
        (4, (4, 5, 6)),
        (7, (7,)),
    ]


def test_skip_equal_to_length_yields_nothing():
    """Skipping the entire list yields no batches."""
    data = list(range(5))
    result = list(batched_iteration_with_skip(data, batch_size=2, skip_count=5))
    assert result == []


def test_skip_larger_than_length_yields_nothing():
    """Skipping past the end of the list yields no batches and does not raise."""
    data = list(range(5))
    result = list(batched_iteration_with_skip(data, batch_size=2, skip_count=100))
    assert result == []


def test_empty_list():
    """An empty input yields no batches regardless of skip_count."""
    assert list(batched_iteration_with_skip([], batch_size=4, skip_count=0)) == []
    assert list(batched_iteration_with_skip([], batch_size=4, skip_count=3)) == []


def test_batch_size_larger_than_remaining():
    """A batch_size exceeding the remaining items yields one full-remainder batch."""
    data = list(range(6))
    result = list(batched_iteration_with_skip(data, batch_size=100, skip_count=2))
    assert result == [(2, (2, 3, 4, 5))]


@pytest.mark.parametrize("batch_size", [0, -1, -10])
def test_invalid_batch_size_raises(batch_size):
    """A batch_size below 1 raises ValueError."""
    with pytest.raises(ValueError, match="batch size must be at least one"):
        list(batched_iteration_with_skip([1, 2, 3], batch_size, skip_count=0))


@pytest.mark.parametrize("skip_count", [-1, -5])
def test_negative_skip_count_raises(skip_count):
    """A negative skip_count raises ValueError."""
    with pytest.raises(ValueError, match="skip_count must be non-negative"):
        list(
            batched_iteration_with_skip([1, 2, 3], batch_size=2, skip_count=skip_count)
        )


def test_returns_tuples_not_lists():
    """Each yielded batch is a tuple, mirroring batched_iteration."""
    _, batch = next(
        batched_iteration_with_skip([1, 2, 3, 4], batch_size=2, skip_count=0)
    )
    assert isinstance(batch, tuple)
