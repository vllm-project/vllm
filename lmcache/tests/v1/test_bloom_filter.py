# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for lmcache.v1.utils.bloom_filter.

Covers add / contains semantics, batch insertion, clear, and the
statistics helpers across both string and integer item types.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.utils.bloom_filter import BloomFilter


class TestSizing:
    def test_default_construction(self) -> None:
        bf = BloomFilter()
        assert bf.expected_elements == 1_000_000
        assert bf.false_positive_rate == 0.01
        assert bf.size > 0
        assert bf.hash_count >= 1
        assert bf.item_count == 0

    def test_smaller_target_rate_yields_larger_bit_array(self) -> None:
        loose = BloomFilter(expected_elements=1000, false_positive_rate=0.1)
        tight = BloomFilter(expected_elements=1000, false_positive_rate=0.001)
        assert tight.size > loose.size
        assert tight.hash_count >= loose.hash_count

    @pytest.mark.parametrize(
        "expected_elements,false_positive_rate,expected_size,expected_hash_count",
        [
            # Hard-coded golden values from the standard Bloom-filter formulas
            # m = -n * ln(p) / (ln 2)^2 and k = (m / n) * ln 2, with the
            # implementation's int()/max(1, ...) rounding applied. Hard-coding
            # avoids the tautology of reproducing the production formula in
            # the test body — these values are independently re-derivable.
            (5000, 0.01, 47925, 6),
            (1000, 0.01, 9585, 6),
            (10000, 0.001, 143775, 9),
        ],
    )
    def test_optimal_sizing_matches_golden_values(
        self,
        expected_elements: int,
        false_positive_rate: float,
        expected_size: int,
        expected_hash_count: int,
    ) -> None:
        bf = BloomFilter(
            expected_elements=expected_elements,
            false_positive_rate=false_positive_rate,
        )
        assert bf.size == expected_size
        assert bf.hash_count == expected_hash_count


class TestContainsString:
    def test_no_false_negatives(self) -> None:
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        items = [f"key-{i}" for i in range(500)]
        for item in items:
            bf.add(item)
        for item in items:
            assert bf.contains(item)
        assert bf.item_count == 500

    def test_empty_filter_contains_nothing(self) -> None:
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)
        assert not bf.contains("anything")
        assert not bf.contains("")

    def test_false_positive_rate_within_reasonable_bound(self) -> None:
        # Honour the design budget loosely: with 1000 inserted and 10000
        # never-inserted items, observed false-positive rate should stay
        # below 5x the target. Inputs are fixed strings hashed with SHA-256,
        # so this test is fully deterministic (no RNG, no flakes).
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        for i in range(1000):
            bf.add(f"in-{i}")
        misses = sum(1 for i in range(10000) if bf.contains(f"out-{i}"))
        observed_rate = misses / 10000
        assert observed_rate < 0.05


class TestContainsInt:
    def test_int_items_round_trip(self) -> None:
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        for n in (0, 1, 2, 255, 65536, 10**18):
            bf.add(n)
            assert bf.contains(n)
        # item_count counts add() calls regardless of duplicates
        assert bf.item_count == 6

    def test_int_zero_round_trips(self) -> None:
        # n=0 exercises the (0).to_bytes(0, ...) edge case in _hashes;
        # call out separately so a regression here is unambiguous.
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        bf.add(0)
        assert bf.contains(0)

    @pytest.mark.parametrize("value", [0, 1, 42, 123456789])
    def test_int_and_str_use_disjoint_hash_inputs(self, value: int) -> None:
        # The contract is structural, not probabilistic: ints are hashed via
        # int.to_bytes(...) while strings are hashed via f"{item}_{i}".encode().
        # The byte streams are disjoint by construction, so contains(str(value))
        # is deterministically False after only add(value) — no RNG involved.
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        bf.add(value)
        assert bf.contains(value)
        assert not bf.contains(str(value))


class TestBatchInsert:
    """Batch insertion takes raw bit positions, not items.

    Tests use hand-picked positions to verify the dedup-by-bitmap behavior
    without relying on the BloomFilter's internal hashing details.
    """

    def test_batch_returns_unique_count_for_distinct_positions(self) -> None:
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        # Three position sets, each touching disjoint bits → all three are new.
        added = bf.add_batch_with_hashes_and_check(
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
        )
        assert added == 3
        assert bf.item_count == 3

    def test_batch_dedups_when_all_bits_already_set(self) -> None:
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        first = bf.add_batch_with_hashes_and_check([[5, 6, 7]])
        # Same positions again — every bit is already set, so it counts as 0.
        second = bf.add_batch_with_hashes_and_check([[5, 6, 7]])
        assert first == 1
        assert second == 0
        assert bf.item_count == 1

    def test_batch_treats_partial_overlap_as_new(self) -> None:
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        bf.add_batch_with_hashes_and_check([[0, 1]])
        # Position 2 is unset → the whole positions list is treated as new.
        added = bf.add_batch_with_hashes_and_check([[0, 1, 2]])
        assert added == 1
        assert bf.item_count == 2


class TestClearAndStats:
    def test_clear_resets_state(self) -> None:
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)
        for i in range(50):
            bf.add(f"k{i}")
        assert bf.get_bit_set() > 0
        bf.clear()
        assert bf.item_count == 0
        assert bf.get_bit_set() == 0
        assert bf.get_fill_rate() == 0.0
        for i in range(50):
            assert not bf.contains(f"k{i}")

    def test_memory_usage_matches_size(self) -> None:
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)
        assert bf.get_memory_usage_bytes() == bf.size // 8

    def test_statistics_shape_and_values(self) -> None:
        bf = BloomFilter(expected_elements=500, false_positive_rate=0.01)
        for i in range(10):
            bf.add(f"k{i}")
        stats = bf.get_statistics()
        assert set(stats.keys()) == {
            "size_mb",
            "hash_count",
            "item_count",
            "bits_set",
            "fill_rate",
            "expected_elements",
            "false_positive_rate",
        }
        assert stats["item_count"] == 10
        assert stats["expected_elements"] == 500
        assert stats["false_positive_rate"] == pytest.approx(0.01)
        assert stats["bits_set"] > 0
        assert 0.0 < stats["fill_rate"] < 1.0
        # Mirror production: size_mb derives from get_memory_usage_bytes(),
        # which uses floor division. Reproducing `bf.size / 8` here only
        # matched production when bf.size happened to be divisible by 8.
        assert stats["size_mb"] == pytest.approx(
            bf.get_memory_usage_bytes() / 1024 / 1024
        )
