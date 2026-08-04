# SPDX-License-Identifier: Apache-2.0

# Standard
import random

# Third Party
import pytest

pytest.importorskip(
    "lmcache.native_storage_ops",
    reason="native_storage_ops extension not built",
)

# First Party
from lmcache.native_storage_ops import Bitmap


class TestBitmapSetClearTest:
    """Test set, clear, and test for various sizes (including non-multiple-of-8)."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 15, 16, 17, 24, 25])
    def test_set_and_test(self, size):
        b = Bitmap(size)
        for i in range(size):
            assert not b.test(i)
        for i in range(0, size, 2):
            b.set(i)
        for i in range(size):
            assert b.test(i) == (i % 2 == 0)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 15, 16, 17])
    def test_clear(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        for i in range(size):
            assert b.test(i)
        for i in range(0, size, 2):
            b.clear(i)
        for i in range(size):
            assert b.test(i) == (i % 2 == 1)

    @pytest.mark.parametrize("size", [1, 5, 9, 12])
    def test_out_of_range_no_op(self, size):
        b = Bitmap(size)
        b.set(size)  # no-op
        b.set(size + 10)  # no-op
        b.clear(size)
        b.clear(size + 10)
        for i in range(size):
            assert not b.test(i)
        assert not b.test(size)
        assert not b.test(size + 1)

    def test_single_bit_size_one(self):
        b = Bitmap(1)
        assert not b.test(0)
        b.set(0)
        assert b.test(0)
        b.clear(0)
        assert not b.test(0)


class TestBitmapSetRange:
    """set_range([start, end)) must match the equivalent per-bit set, including
    single-byte, head/middle/tail spans, clamping, and empty ranges."""

    @pytest.mark.parametrize(
        "size,start,end",
        [
            (16, 3, 3),  # empty range -> no-op
            (16, 0, 0),  # empty at 0
            (16, 2, 5),  # within one byte
            (16, 0, 8),  # exactly one full byte
            (16, 0, 16),  # whole bitmap
            (24, 3, 20),  # head + middle byte + tail
            (24, 8, 16),  # exactly the middle byte
            (10, 5, 100),  # end clamped to size
            (10, 9, 10),  # last bit only
            (10, 100, 200),  # fully out of range -> no-op
        ],
    )
    def test_set_range_matches_per_bit(self, size, start, end):
        ranged = Bitmap(size)
        ranged.set_range(start, end)

        per_bit = Bitmap(size)
        for i in range(start, min(end, size)):
            per_bit.set(i)

        assert ranged.get_indices_list() == per_bit.get_indices_list()

    def test_set_range_accumulates(self):
        b = Bitmap(32)
        b.set_range(2, 6)
        b.set_range(10, 12)
        assert b.get_indices_list() == [2, 3, 4, 5, 10, 11]

    def test_set_range_random_against_per_bit(self):
        rng = random.Random(0)
        for _ in range(500):
            size = rng.randint(0, 40)
            start = rng.randint(0, size + 4)
            end = rng.randint(0, size + 4)
            ranged = Bitmap(size)
            ranged.set_range(start, end)
            per_bit = Bitmap(size)
            for i in range(start, min(end, size)):
                per_bit.set(i)
            assert ranged.get_indices_list() == per_bit.get_indices_list()


class TestBitmapPopcount:
    """Test popcount (count of set bits), especially for size not multiple of 8."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 15, 16, 17, 24, 25])
    def test_popcount_all_zeros(self, size):
        b = Bitmap(size)
        assert b.popcount() == 0

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 15, 16, 17])
    def test_popcount_all_ones(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        assert b.popcount() == size

    @pytest.mark.parametrize("size", [1, 5, 9, 12, 20])
    def test_popcount_partial(self, size):
        b = Bitmap(size)
        for i in range(0, size, 2):
            b.set(i)
        expected = (size + 1) // 2
        assert b.popcount() == expected, f"size={size}"

    @pytest.mark.parametrize("size", [9, 12, 20])
    def test_popcount_random(self, size):
        b = Bitmap(size)
        one_counts = 5
        one_positions = random.sample(range(size), one_counts)
        for i in one_positions:
            b.set(i)
        assert b.popcount() == one_counts


class TestBitmapClzClo:
    """Test count leading zeros (clz) and count leading ones (clo)."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 15, 16, 17])
    def test_clz_all_zeros(self, size):
        b = Bitmap(size)
        assert b.count_leading_zeros() == size

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_clz_first_bit_set(self, size):
        b = Bitmap(size)
        b.set(0)
        assert b.count_leading_zeros() == 0

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10])
    def test_clz_second_bit_set(self, size):
        b = Bitmap(size)
        b.set(1)
        assert b.count_leading_zeros() == 1

    def test_clz_middle_bits_set(self):
        # size=9: bits 0..8; set 3,4,5 -> leading zeros = 3
        b = Bitmap(9)
        b.set(3)
        b.set(4)
        b.set(5)
        assert b.count_leading_zeros() == 3

    def test_clz_last_bit_only_size_9(self):
        b = Bitmap(9)
        b.set(8)  # only last bit (index 8)
        assert b.count_leading_zeros() == 8

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_clo_all_ones(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        assert b.count_leading_ones() == size

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10])
    def test_clo_all_zeros(self, size):
        b = Bitmap(size)
        assert b.count_leading_ones() == 0

    def test_clo_first_zero_at_3(self):
        b = Bitmap(9)
        b.set(0)
        b.set(1)
        b.set(2)
        # bits 3..8 are 0
        assert b.count_leading_ones() == 3

    def test_clo_partial_byte_size_10(self):
        # 10 bits: all set -> clo = 10
        b = Bitmap(10)
        for i in range(10):
            b.set(i)
        assert b.count_leading_ones() == 10

    def test_clo_partial_byte_first_zero_at_5(self):
        b = Bitmap(10)
        for i in range(5):
            b.set(i)
        assert b.count_leading_ones() == 5


class TestBitmapAnd:
    """Test bitwise AND between two bitmaps."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 16, 17])
    def test_and_same_size_all_overlap(self, size):
        a = Bitmap(size)
        b = Bitmap(size)
        for i in range(0, size, 2):
            a.set(i)
        for i in range(1, size, 2):
            b.set(i)
        c = a & b
        assert c.popcount() == 0
        for i in range(size):
            assert not c.test(i)

    @pytest.mark.parametrize("size", [1, 9, 10])
    def test_and_same_size_intersection(self, size):
        a = Bitmap(size)
        b = Bitmap(size)
        for i in range(size):
            a.set(i)
        for i in range(0, size, 2):
            b.set(i)
        c = a & b
        assert c.popcount() == (size + 1) // 2
        for i in range(size):
            assert c.test(i) == (i % 2 == 0)

    def test_and_different_sizes_result_truncated(self):
        a = Bitmap(10)
        b = Bitmap(5)
        for i in range(10):
            a.set(i)
        for i in range(5):
            b.set(i)
        c = a & b
        # Result is min(10,5)=5 bits, all set
        assert c.popcount() == 5
        for i in range(5):
            assert c.test(i)

    def test_and_different_sizes_other_longer(self):
        a = Bitmap(5)
        b = Bitmap(10)
        for i in range(5):
            a.set(i)
        for i in range(10):
            b.set(i)
        c = a & b
        assert c.popcount() == 5
        for i in range(5):
            assert c.test(i)


class TestBitmapToString:
    """Test string representation (bit 0 = leftmost character)."""

    def test_to_string_empty_zero_bits(self):
        # Size 0 might not be allowed; skip or use size 1
        b = Bitmap(1)
        b.clear(0)
        assert "0" in str(b)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10])
    def test_to_string_matches_set_bits(self, size):
        b = Bitmap(size)
        for i in range(0, size, 2):
            b.set(i)
        s = str(b)
        print(s)
        assert len(s) == size
        for i in range(size):
            assert s[i] == "1" if (i % 2 == 0) else "0"

    def test_repr_calls_to_string(self):
        b = Bitmap(3)
        b.set(1)
        r = repr(b)
        assert "1" in r and "0" in r


class TestBitmapOr:
    """Test bitwise OR between two bitmaps."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 16, 17])
    def test_or_same_size_no_overlap(self, size):
        a = Bitmap(size)
        b = Bitmap(size)
        for i in range(0, size, 2):
            a.set(i)
        for i in range(1, size, 2):
            b.set(i)
        c = a | b
        assert c.popcount() == size
        for i in range(size):
            assert c.test(i)

    @pytest.mark.parametrize("size", [1, 9, 10])
    def test_or_same_size_with_overlap(self, size):
        a = Bitmap(size)
        b = Bitmap(size)
        for i in range(size):
            a.set(i)
        for i in range(0, size, 2):
            b.set(i)
        c = a | b
        assert c.popcount() == size

    def test_or_different_sizes_result_truncated(self):
        a = Bitmap(10)
        b = Bitmap(5)
        for i in range(0, 10, 2):
            a.set(i)
        for i in range(1, 5, 2):
            b.set(i)
        c = a | b
        # Result is min(10,5)=5 bits
        assert c.popcount() == 5
        for i in range(5):
            assert c.test(i)

    def test_or_both_empty(self):
        a = Bitmap(8)
        b = Bitmap(8)
        c = a | b
        assert c.popcount() == 0


class TestBitmapGetIndices:
    """Test get_indices_list and get_indices_set."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_get_indices_list_all_zeros(self, size):
        b = Bitmap(size)
        assert b.get_indices_list() == []

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_get_indices_list_all_ones(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        assert b.get_indices_list() == list(range(size))

    @pytest.mark.parametrize("size", [9, 10, 17, 25])
    def test_get_indices_list_even_bits(self, size):
        b = Bitmap(size)
        expected = list(range(0, size, 2))
        for i in expected:
            b.set(i)
        assert b.get_indices_list() == expected

    @pytest.mark.parametrize("size", [9, 12, 20])
    def test_get_indices_list_random(self, size):
        b = Bitmap(size)
        positions = sorted(random.sample(range(size), 5))
        for i in positions:
            b.set(i)
        assert b.get_indices_list() == positions

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_get_indices_set_all_zeros(self, size):
        b = Bitmap(size)
        assert b.get_indices_set() == set()

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_get_indices_set_all_ones(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        assert b.get_indices_set() == set(range(size))

    @pytest.mark.parametrize("size", [9, 12, 20])
    def test_get_indices_set_random(self, size):
        b = Bitmap(size)
        positions = random.sample(range(size), 5)
        for i in positions:
            b.set(i)
        assert b.get_indices_set() == set(positions)

    def test_get_indices_list_and_set_consistent(self):
        b = Bitmap(17)
        for i in [0, 3, 8, 15, 16]:
            b.set(i)
        assert set(b.get_indices_list()) == b.get_indices_set()


class TestBitmapGather:
    """Test gather operation."""

    def test_gather_basic(self):
        b = Bitmap(5)
        b.set(0)
        b.set(2)
        b.set(4)
        items = ["a", "b", "c", "d", "e"]
        assert b.gather(items) == ["a", "c", "e"]

    def test_gather_all_set(self):
        b = Bitmap(4)
        for i in range(4):
            b.set(i)
        items = [10, 20, 30, 40]
        assert b.gather(items) == [10, 20, 30, 40]

    def test_gather_none_set(self):
        b = Bitmap(3)
        items = ["x", "y", "z"]
        assert b.gather(items) == []

    def test_gather_mixed_types(self):
        b = Bitmap(4)
        b.set(1)
        b.set(3)
        items = [1, "two", 3.0, None]
        assert b.gather(items) == ["two", None]

    @pytest.mark.parametrize("size", [9, 10, 17])
    def test_gather_preserves_order(self, size):
        b = Bitmap(size)
        positions = sorted(random.sample(range(size), 5))
        for i in positions:
            b.set(i)
        items = list(range(size))
        assert b.gather(items) == positions

    def test_gather_single_element(self):
        b = Bitmap(1)
        b.set(0)
        assert b.gather(["only"]) == ["only"]

    def test_gather_accepts_tuple(self):
        # gather takes any sequence, not just list (e.g. l2_orig_indices tuple).
        b = Bitmap(5)
        b.set(1)
        b.set(3)
        assert b.gather((10, 11, 12, 13, 14)) == [11, 13]


class TestBitmapInvert:
    """Test bitwise NOT (~) operation."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 16, 17])
    def test_invert_all_zeros_gives_all_ones(self, size):
        b = Bitmap(size)
        inv = ~b
        assert inv.popcount() == size
        for i in range(size):
            assert inv.test(i)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 16, 17])
    def test_invert_all_ones_gives_all_zeros(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        inv = ~b
        assert inv.popcount() == 0
        for i in range(size):
            assert not inv.test(i)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17, 25])
    def test_invert_flips_each_bit(self, size):
        b = Bitmap(size)
        for i in range(0, size, 2):
            b.set(i)
        inv = ~b
        for i in range(size):
            assert inv.test(i) == (i % 2 == 1)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 17])
    def test_double_invert_is_identity(self, size):
        b = Bitmap(size)
        positions = [i for i in range(size) if i % 3 == 0]
        for i in positions:
            b.set(i)
        result = ~(~b)
        for i in range(size):
            assert result.test(i) == b.test(i)

    def test_invert_does_not_mutate_original(self):
        b = Bitmap(10)
        b.set(0)
        b.set(5)
        _ = ~b
        assert b.popcount() == 2
        assert b.test(0)
        assert b.test(5)

    @pytest.mark.parametrize("size", [9, 10, 17])
    def test_invert_popcount_complement(self, size):
        b = Bitmap(size)
        for i in range(0, size, 3):
            b.set(i)
        inv = ~b
        assert b.popcount() + inv.popcount() == size

    def test_invert_and_original_gives_zero(self):
        b = Bitmap(10)
        for i in [1, 3, 5, 7, 9]:
            b.set(i)
        result = b & ~b
        assert result.popcount() == 0

    def test_invert_or_original_gives_all_ones(self):
        b = Bitmap(10)
        for i in [1, 3, 5, 7, 9]:
            b.set(i)
        result = b | ~b
        assert result.popcount() == 10


class TestBitmapPrefixConstructor:
    """Test Bitmap(size, prefix_bits) constructor."""

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 16, 17, 25])
    def test_prefix_zero(self, size):
        b = Bitmap(size, 0)
        assert b.popcount() == 0

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 10, 16, 17, 25])
    def test_prefix_all(self, size):
        b = Bitmap(size, size)
        assert b.popcount() == size
        assert b.count_leading_ones() == size

    @pytest.mark.parametrize("size", [8, 9, 10, 16, 17, 25])
    def test_prefix_partial(self, size):
        prefix = size // 2
        b = Bitmap(size, prefix)
        assert b.popcount() == prefix
        assert b.count_leading_ones() == prefix
        for i in range(prefix):
            assert b.test(i)
        for i in range(prefix, size):
            assert not b.test(i)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 16, 17])
    def test_prefix_one(self, size):
        b = Bitmap(size, 1)
        assert b.popcount() == 1
        assert b.test(0)
        for i in range(1, size):
            assert not b.test(i)

    def test_prefix_exceeds_size_clamped(self):
        b = Bitmap(5, 100)
        assert b.popcount() == 5
        assert b.count_leading_ones() == 5

    @pytest.mark.parametrize("size", [8, 16, 24])
    def test_prefix_exact_byte_boundary(self, size):
        b = Bitmap(size, 8)
        assert b.popcount() == 8
        assert b.count_leading_ones() == 8
        for i in range(8):
            assert b.test(i)
        for i in range(8, size):
            assert not b.test(i)

    @pytest.mark.parametrize(
        "size,prefix",
        [(9, 3), (10, 5), (17, 9), (25, 13)],
    )
    def test_prefix_matches_manual_set(self, size, prefix):
        """Bitmap(size, prefix) should be identical to setting bits 0..prefix-1."""
        a = Bitmap(size, prefix)
        b = Bitmap(size)
        for i in range(prefix):
            b.set(i)
        assert a.popcount() == b.popcount()
        assert a.get_indices_list() == b.get_indices_list()
        assert str(a) == str(b)

    def test_prefix_zero_size(self):
        b = Bitmap(0, 0)
        assert b.popcount() == 0


class TestBitmapNonMultipleOfEight:
    """Focused tests when length is not a multiple of 8."""

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 9, 10, 11, 15, 17, 23, 25])
    def test_roundtrip_set_test_all_positions(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
            assert b.test(i)
            b.clear(i)
            assert not b.test(i)
            b.set(i)

    @pytest.mark.parametrize("size", [9, 10, 12, 17])
    def test_popcount_only_low_bits_counted_in_last_byte(self, size):
        b = Bitmap(size)
        for i in range(size):
            b.set(i)
        assert b.popcount() == size

    @pytest.mark.parametrize("size", [9, 10, 12])
    def test_clz_clo_consistent_with_test(self, size):
        b = Bitmap(size)
        b.set(size - 1)  # only last bit set
        assert b.count_leading_zeros() == size - 1
        assert b.count_leading_ones() == 0

        b2 = Bitmap(size)
        for i in range(size - 1):
            b2.set(i)
        assert b2.count_leading_ones() == size - 1
        assert b2.count_leading_zeros() == 0


class TestBatchedSet:
    """Bitmap.batched_set(indices): set every listed bit; positions >= size
    are ignored (bounds-checked, like set())."""

    def test_sets_listed_bits(self):
        b = Bitmap(8)
        b.batched_set([1, 3, 7])
        assert b.get_indices_set() == {1, 3, 7}

    def test_accumulates_with_existing(self):
        b = Bitmap(8)
        b.set(0)
        b.batched_set([2, 4])
        assert b.get_indices_set() == {0, 2, 4}

    def test_out_of_range_ignored(self):
        b = Bitmap(5)
        b.batched_set([1, 5, 99])  # 5 and 99 are >= size -> dropped
        assert b.get_indices_set() == {1}

    def test_empty_is_noop(self):
        b = Bitmap(4)
        b.set(2)
        b.batched_set([])
        assert b.get_indices_set() == {2}

    def test_duplicates_idempotent(self):
        b = Bitmap(4)
        b.batched_set([1, 1, 1])
        assert b.get_indices_set() == {1}

    def test_accepts_tuple(self):
        b = Bitmap(4)
        b.batched_set((0, 3))
        assert b.get_indices_set() == {0, 3}

    def test_matches_python_reference(self):
        rng = random.Random(1234)
        for _ in range(50):
            size = rng.randint(1, 64)
            indices = rng.sample(range(size), k=rng.randint(0, size))
            b = Bitmap(size)
            b.batched_set(indices)
            assert b.get_indices_set() == set(indices)

    def test_combine_found_pattern(self):
        # The _combine_found composition: seed L1 indices, then batched_set the
        # L2 set bits gathered through their original-position map.
        l2 = Bitmap(3)  # L2 result over the 3 L2-submitted keys
        l2.set(0)
        l2.set(2)
        l2_orig = (3, 5, 7)  # L2-local i -> original position
        found = Bitmap(10)
        found.batched_set([0, 1, 2])  # L1 prefix
        found.batched_set(l2.gather(l2_orig))  # -> {3, 7}
        assert found.get_indices_set() == {0, 1, 2, 3, 7}
