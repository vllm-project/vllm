# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest
import torch

pytest.importorskip(
    "lmcache.native_storage_ops",
    reason="native_storage_ops extension not built",
)

# First Party
from lmcache.native_storage_ops import ParallelPatternMatcher, RangePatternMatcher


def assert_torch():
    """Test that torch is available and can be imported."""
    assert torch.__version__ is not None, "Torch should be available"


class TestParallelPatternMatcherBasic:
    """Test basic functionality of ParallelPatternMatcher."""

    def test_example_from_spec(self):
        """Test the exact example from the specification."""
        pattern = [1, 2, 3]
        data = [1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [3, 8], f"Expected [3, 8] but got {result}"

    def test_no_matches(self):
        """Test when pattern is not found in data."""
        pattern = [5, 6, 7]
        data = [1, 2, 3, 4, 1, 2, 3, 4]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == []

    def test_single_match(self):
        """Test when pattern appears once."""
        pattern = [7, 8, 9]
        data = [1, 2, 3, 7, 8, 9, 10, 11]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [3]

    def test_pattern_at_start(self):
        """Test when pattern is at the beginning of data."""
        pattern = [1, 2, 3]
        data = [1, 2, 3, 4, 5, 6]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0]

    def test_pattern_at_end(self):
        """Test when pattern is at the end of data."""
        pattern = [4, 5, 6]
        data = [1, 2, 3, 4, 5, 6]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [3]

    def test_multiple_matches(self):
        """Test when pattern appears multiple times."""
        pattern = [1, 2]
        data = [1, 2, 3, 1, 2, 4, 1, 2, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0, 3, 6]

    def test_overlapping_patterns(self):
        """Test when patterns overlap."""
        pattern = [1, 1]
        data = [1, 1, 1, 1, 2]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Overlapping patterns at positions 0, 1, 2
        assert result == [0, 1, 2]

    def test_single_element_pattern(self):
        """Test with a single-element pattern."""
        pattern = [5]
        data = [1, 2, 5, 3, 5, 4, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [2, 4, 6]

    def test_pattern_equals_data(self):
        """Test when pattern is exactly the same as data."""
        pattern = [1, 2, 3, 4, 5]
        data = [1, 2, 3, 4, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0]

    def test_pattern_longer_than_data(self):
        """Test when pattern is longer than data."""
        pattern = [1, 2, 3, 4, 5]
        data = [1, 2, 3]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == []

    def test_empty_data(self):
        """Test with empty data."""
        pattern = [1, 2, 3]
        data = []

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == []

    def test_large_pattern(self):
        """Test with a large pattern."""
        pattern = list(range(10))
        data = [0] + list(range(10)) + [99] + list(range(10)) + [88]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [1, 12]


class TestParallelPatternMatcherConsistency:
    """Test consistency of pattern matching."""

    def test_repeated_matches_consistent(self):
        """Test that repeated calls produce consistent results."""
        pattern = [1, 2, 3]
        data = [1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3]

        matcher = ParallelPatternMatcher(pattern)

        # Call match multiple times
        results = [matcher.match(data) for _ in range(5)]

        # All results should be the same
        for result in results:
            assert result == [3, 8]

    def test_multiple_matchers_same_result(self):
        """Test that multiple matcher instances produce the same result."""
        pattern = [5, 6, 7]
        data = list(range(100)) + [5, 6, 7] + list(range(50)) + [5, 6, 7]

        results = []
        for _ in range(5):
            matcher = ParallelPatternMatcher(pattern)
            result = matcher.match(data)
            results.append(result)

        # All results should be the same
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Results differ: {results[0]} vs {results[i]}"
            )


class TestParallelPatternMatcherEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_numbers(self):
        """Test with negative numbers in pattern and data."""
        pattern = [-1, -2, -3]
        data = [1, 2, -1, -2, -3, 4, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [2]

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative numbers."""
        pattern = [1, -1, 2]
        data = [0, 1, -1, 2, 3, 1, -1, 2, 4]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [1, 5]

    def test_zeros_in_pattern(self):
        """Test with zeros in the pattern."""
        pattern = [0, 0, 0]
        data = [1, 2, 0, 0, 0, 3, 0, 0, 0]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [2, 6]

    def test_large_numbers(self):
        """Test with large numbers."""
        pattern = [1000000, 2000000, 3000000]
        data = [1, 1000000, 2000000, 3000000, 2, 1000000, 2000000, 3000000]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [1, 5]

    def test_repeating_pattern_in_repeating_data(self):
        """Test repeating pattern in repeating data."""
        pattern = [1, 2]
        data = [1, 2, 1, 2, 1, 2, 1, 2]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0, 2, 4, 6]

    def test_pattern_at_every_position(self):
        """Test when pattern could match at multiple overlapping positions."""
        pattern = [1, 1, 1]
        data = [1, 1, 1, 1, 1]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Pattern matches at positions 0, 1, 2
        assert result == [0, 1, 2]


class TestParallelPatternMatcherThreadSafety:
    """Test thread safety when using the same matcher from multiple threads."""

    def test_concurrent_matches(self):
        """Test that multiple threads can call match() concurrently."""
        pattern = [1, 2, 3]
        data = [1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3]

        matcher = ParallelPatternMatcher(pattern)

        def do_match():
            result = matcher.match(data)
            assert result == [3, 8]

        num_threads = 10
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=do_match)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def test_concurrent_different_data(self):
        """Test concurrent matches with different data."""
        pattern = [5, 6, 7]
        matcher = ParallelPatternMatcher(pattern)

        data_sets = [
            ([5, 6, 7, 1, 2, 3], [0]),
            ([1, 2, 5, 6, 7, 8], [2]),
            ([1, 2, 3, 4, 5, 6], []),
            ([5, 6, 7, 5, 6, 7], [0, 3]),
        ]

        results = [None] * len(data_sets)

        def do_match(idx, data, expected):
            result = matcher.match(data)
            assert result == expected, f"Expected {expected} but got {result}"
            results[idx] = result

        threads = []
        for idx, (data, expected) in enumerate(data_sets):
            t = threading.Thread(target=do_match, args=(idx, data, expected))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify all results
        for idx, (_, expected) in enumerate(data_sets):
            assert results[idx] == expected


class TestParallelPatternMatcherPerformance:
    """Test performance characteristics."""

    def test_large_data(self):
        """Test with large data."""
        pattern = [42, 43, 42]
        data = list(range(10000)) + [42, 43, 42] + list(range(5000))

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [10000]

    def test_many_matches_in_large_data(self):
        """Test with many pattern occurrences in large data."""
        pattern = [1, 2, 3]
        # Create data with pattern appearing every 10 elements
        data = []
        for i in range(1000):
            if i % 10 == 0:
                data.extend([1, 2, 3])
            else:
                data.extend([4, 5, 6])

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Verify we found all occurrences
        for idx in result:
            assert data[idx : idx + 3] == [1, 2, 3], f"Invalid match at position {idx}"

        # Should have ~100 matches
        assert len(result) > 90

    def test_stress_test_parallel_execution(self):
        """Stress test with multiple matchers running in parallel."""
        pattern = [7, 8, 9]
        data = list(range(1000)) + [7, 8, 9] + list(range(500))

        def worker():
            matcher = ParallelPatternMatcher(pattern)
            result = matcher.match(data)
            assert 1000 in result

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(100)]
            for f in futures:
                f.result()  # Will raise if any thread failed


class TestParallelPatternMatcherResultSorted:
    """Test that results are always sorted."""

    def test_results_are_sorted(self):
        """Test that matches are returned in sorted order."""
        pattern = [1, 2]
        # Create data where pattern appears in various positions
        data = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Verify sorted
        assert result == sorted(result)

        # Verify correctness
        assert result == [1, 4, 8]

    def test_results_sorted_multiple_matches(self):
        """Test that results are sorted with multiple matches."""
        pattern = [5]
        data = [5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == sorted(result)
        assert result == [0, 2, 4, 6, 8, 10]


class TestRangePatternMatcherBasic:
    """Test basic functionality of RangePatternMatcher."""

    def test_example_from_spec(self):
        """Test the exact example from the specification."""
        start = [1, 2]
        end = [3, 4]
        data = [1, 2, 0, 3, 4, 0, 3, 4, 1, 2, 0, 0, 3, 4]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 5), (8, 14)], (
            f"Expected [(0, 5), (8, 14)] but got {result}"
        )

    def test_single_range(self):
        """Test when there's a single range in the data."""
        start = [1, 2]
        end = [5, 6]
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(1, 7)]

    def test_no_matches(self):
        """Test when no start pattern is found."""
        start = [10, 11]
        end = [20, 21]
        data = [1, 2, 3, 4, 5, 6, 7, 8]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_start_without_end(self):
        """Test when start pattern is found but no end pattern follows."""
        start = [1, 2]
        end = [9, 10]
        data = [1, 2, 3, 4, 5, 6, 7, 8]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_end_without_start(self):
        """Test when end pattern exists but no start pattern before it."""
        start = [1, 2]
        end = [7, 8]
        data = [0, 3, 4, 5, 6, 7, 8, 9]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_multiple_ranges(self):
        """Test multiple ranges in the data."""
        start = [1]
        end = [2]
        data = [1, 0, 2, 1, 0, 0, 2, 1, 2]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 3), (3, 7), (7, 9)]

    def test_minimal_range_selection(self):
        """Test that the first end pattern is selected (minimal range)."""
        start = [1]
        end = [2]
        data = [1, 0, 2, 0, 2, 0, 2]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        # Should match to the first occurrence of end pattern at position 2
        assert result == [(0, 3)]

    def test_adjacent_start_end(self):
        """Test when start and end patterns are adjacent."""
        start = [1, 2]
        end = [3, 4]
        data = [0, 1, 2, 3, 4, 5]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(1, 5)]

    def test_start_equals_data(self):
        """Test when start pattern equals all the data."""
        start = [1, 2, 3, 4]
        end = [5, 6]
        data = [1, 2, 3, 4]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_single_element_patterns(self):
        """Test with single-element patterns."""
        start = [1]
        end = [5]
        data = [1, 2, 3, 4, 5, 6, 1, 2, 5]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 5), (6, 9)]


class TestRangePatternMatcherEdgeCases:
    """Test edge cases and boundary conditions for RangePatternMatcher."""

    def test_empty_data(self):
        """Test with empty data."""
        start = [1, 2]
        end = [3, 4]
        data = []

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_data_too_short(self):
        """Test when data is shorter than start + end patterns."""
        start = [1, 2, 3]
        end = [7, 8, 9]
        data = [1, 2]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_pattern_at_boundaries(self):
        """Test patterns at the start and end of data."""
        start = [1]
        end = [9]
        data = [1, 2, 3, 4, 9]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 5)]

    def test_negative_numbers(self):
        """Test with negative numbers."""
        start = [-1, -2]
        end = [-5, -6]
        data = [0, -1, -2, -3, -4, -5, -6, 10]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(1, 7)]

    def test_zeros_in_patterns(self):
        """Test with zeros in patterns."""
        start = [0, 0]
        end = [1, 1]
        data = [0, 0, 5, 5, 1, 1, 0, 0, 1, 1]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 6), (6, 10)]

    def test_large_numbers(self):
        """Test with large numbers."""
        start = [1000000]
        end = [2000000]
        data = [1000000, 999, 2000000, 1000000, 2000000]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 3), (3, 5)]


class TestRangePatternMatcherConsistency:
    """Test consistency of range pattern matching."""

    def test_repeated_matches_consistent(self):
        """Test that repeated calls produce consistent results."""
        start = [1, 2]
        end = [3, 4]
        data = [1, 2, 0, 3, 4, 0, 3, 4, 1, 2, 0, 0, 3, 4]

        matcher = RangePatternMatcher(start, end)

        # Call match multiple times
        results = [matcher.match(data) for _ in range(5)]

        # All results should be the same
        for result in results:
            assert result == [(0, 5), (8, 14)]

    def test_multiple_matchers_same_result(self):
        """Test that multiple matcher instances produce the same result."""
        start = [5, 6]
        end = [7, 8]
        data = [5, 6, 0, 7, 8, 1, 5, 6, 7, 8]

        results = []
        for _ in range(5):
            matcher = RangePatternMatcher(start, end)
            result = matcher.match(data)
            results.append(result)

        # All results should be the same
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Results differ: {results[0]} vs {results[i]}"
            )


class TestRangePatternMatcherComplexScenarios:
    """Test complex scenarios for RangePatternMatcher."""

    def test_nested_like_patterns(self):
        """Test when a new start pattern appears before the first end."""
        start = [1]
        end = [9]
        data = [1, 2, 1, 3, 9, 5, 9]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        # First start at 0, first end at 4 -> (0, 5)
        # Second start at 2 is "inside" the first range, but we already consumed it
        # Continue from 5, no more start patterns
        # Wait, after finding (0, 5), we continue from position 5
        # At position 5, data[5] = 5, not a start
        # At position 6, data[6] = 9, not a start
        # No more ranges
        # But wait, what about the start at position 2?
        # The algorithm processes left to right, so:
        # - Find start at 0, find end at 4, add (0, 5), continue from 5
        # - Check position 5, 6: no starts found
        # So result is [(0, 5)]
        # Actually, I need to reconsider. Let me trace through the algorithm:
        # i=0: start matches, search from 1, end at 4, add (0,5), i=5
        # i=5: data[5]=5, no match, i=6
        # i=6: data[6]=9, no match, i=7
        # i=7: out of bounds
        # Result: [(0, 5)]
        assert result == [(0, 5)]

    def test_multiple_ends_after_start(self):
        """Test multiple end patterns after a start (should pick first)."""
        start = [1, 2]
        end = [5]
        data = [1, 2, 3, 5, 4, 5, 6, 5, 7]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        # Start at 0, first end at 3, range (0, 4)
        assert result == [(0, 4)]

    def test_consecutive_ranges(self):
        """Test consecutive non-overlapping ranges."""
        start = [1]
        end = [2]
        data = [1, 2, 1, 2, 1, 2]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 2), (2, 4), (4, 6)]

    def test_long_data_with_sparse_patterns(self):
        """Test long data with patterns far apart."""
        start = [100]
        end = [200]
        data = [0] * 50 + [100] + [1] * 50 + [200] + [0] * 50 + [100] + [2] * 30 + [200]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(50, 102), (152, 184)]

    def test_pattern_longer_than_data(self):
        """Test when patterns are longer than data."""
        start = [1, 2, 3, 4, 5]
        end = [6, 7]
        data = [1, 2, 3]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == []

    def test_alternating_start_end(self):
        """Test alternating start and end patterns."""
        start = [1]
        end = [2]
        data = [1, 2, 1, 2, 1, 2, 1, 2]

        matcher = RangePatternMatcher(start, end)
        result = matcher.match(data)

        assert result == [(0, 2), (2, 4), (4, 6), (6, 8)]
