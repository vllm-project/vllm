# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for percentile-based tracking in distillation capture."""

import threading

import numpy as np
import pytest

from vllm.v1.spec_decode.capture.percentile_tracker import PercentileTracker


class TestPercentileTracker:
    """Test PercentileTracker functionality."""

    def test_initialization(self):
        """Test percentile tracker initialization."""
        tracker = PercentileTracker(percentile=10.0, window_size=1000, min_samples=100)
        assert tracker.percentile == 10.0
        assert tracker.window_size == 1000
        assert tracker.min_samples == 100
        assert len(tracker.acceptance_window) == 0

    def test_observe_adds_to_window(self):
        """Test that observe adds values to the window."""
        tracker = PercentileTracker()
        tracker.observe(1.5)
        tracker.observe(2.0)
        tracker.observe(1.2)

        assert len(tracker.acceptance_window) == 3
        assert list(tracker.acceptance_window) == [1.5, 2.0, 1.2]

    def test_window_size_limit(self):
        """Test that window respects max size."""
        tracker = PercentileTracker(window_size=10)

        # Add more than window_size values
        for i in range(20):
            tracker.observe(float(i))

        # Should only keep last 10
        assert len(tracker.acceptance_window) == 10
        assert list(tracker.acceptance_window) == list(range(10, 20))

    def test_warmup_period_conservative_threshold(self):
        """Test that during warmup, conservative threshold is used."""
        tracker = PercentileTracker(min_samples=100)

        # During warmup (< min_samples), should use conservative threshold of 1.5
        assert tracker.should_log(1.0) is True  # Below 1.5
        assert tracker.should_log(1.4) is True  # Below 1.5
        assert tracker.should_log(1.6) is False  # Above 1.5
        assert tracker.should_log(2.0) is False  # Above 1.5

    def test_percentile_calculation_after_warmup(self):
        """Test percentile calculation after warmup period."""
        tracker = PercentileTracker(percentile=10.0, min_samples=10)

        # Add 100 values uniformly distributed from 1.0 to 2.0
        values = np.linspace(1.0, 2.0, 100)
        for val in values:
            tracker.observe(val)

        # 10th percentile of uniform [1.0, 2.0] should be around 1.1
        # Values below this should be logged
        assert tracker.should_log(1.05) is True
        assert tracker.should_log(1.15) is False
        assert tracker.should_log(1.5) is False

    def test_observe_and_check_atomic(self):
        """Test that observe_and_check is atomic."""
        tracker = PercentileTracker(percentile=50.0, min_samples=10)

        # Add some initial values
        for i in range(20):
            tracker.observe(float(i))

        # observe_and_check should add value and check in one operation
        initial_len = len(tracker.acceptance_window)
        result = tracker.observe_and_check(5.0)

        assert len(tracker.acceptance_window) == initial_len + 1
        assert isinstance(result, bool)

    def test_observe_and_check_warmup(self):
        """Test observe_and_check during warmup period."""
        tracker = PercentileTracker(min_samples=100)

        # During warmup, should use conservative threshold
        assert tracker.observe_and_check(1.0) is True
        assert tracker.observe_and_check(1.4) is True
        assert tracker.observe_and_check(1.6) is False

    def test_observe_and_check_after_warmup(self):
        """Test observe_and_check after warmup."""
        tracker = PercentileTracker(percentile=25.0, min_samples=10)

        # Add values to complete warmup
        for i in range(1, 21):
            tracker.observe(float(i))

        # 25th percentile of [1..20] is 5.75
        assert tracker.observe_and_check(3.0) is True  # Below 25th percentile
        assert tracker.observe_and_check(10.0) is False  # Above 25th percentile

    def test_get_stats_empty(self):
        """Test get_stats with no observations."""
        tracker = PercentileTracker()
        stats = tracker.get_stats()

        assert stats["num_samples"] == 0
        assert stats["percentile_threshold"] is None
        assert stats["mean_acceptance"] is None
        assert stats["min_acceptance"] is None
        assert stats["max_acceptance"] is None

    def test_get_stats_with_data(self):
        """Test get_stats with observations."""
        tracker = PercentileTracker(percentile=10.0, min_samples=5)

        values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for val in values:
            tracker.observe(val)

        stats = tracker.get_stats()

        assert stats["num_samples"] == len(values)
        assert stats["mean_acceptance"] == pytest.approx(3.0, rel=0.01)
        assert stats["min_acceptance"] == 1.0
        assert stats["max_acceptance"] == 5.0
        assert stats["p25"] == pytest.approx(2.0, rel=0.1)
        assert stats["p50"] == pytest.approx(3.0, rel=0.1)
        assert stats["p75"] == pytest.approx(4.0, rel=0.1)
        assert stats["percentile_threshold"] is not None

    def test_reset(self):
        """Test reset clears all observations."""
        tracker = PercentileTracker()

        # Add some observations
        for i in range(10):
            tracker.observe(float(i))

        assert len(tracker.acceptance_window) > 0

        # Reset
        tracker.reset()

        assert len(tracker.acceptance_window) == 0
        stats = tracker.get_stats()
        assert stats["num_samples"] == 0

    def test_cache_invalidation(self):
        """Test that cache is invalidated periodically."""
        tracker = PercentileTracker(percentile=50.0, min_samples=5)

        # Add enough samples to trigger cache updates
        for i in range(25):
            tracker.observe(float(i))

        # Cache should be invalidated every 10 observations
        # This is tested indirectly by checking that percentile updates
        stats1 = tracker.get_stats()
        threshold1 = stats1["percentile_threshold"]

        # Add more observations that change the distribution
        for i in range(100, 110):
            tracker.observe(float(i))

        stats2 = tracker.get_stats()
        threshold2 = stats2["percentile_threshold"]

        # Threshold should have changed due to new observations
        assert threshold2 != threshold1

    def test_thread_safety(self):
        """Test that percentile tracker is thread-safe."""
        tracker = PercentileTracker(percentile=50.0, min_samples=10)
        num_threads = 10
        observations_per_thread = 100

        def worker(start_val):
            for i in range(observations_per_thread):
                tracker.observe_and_check(start_val + i * 0.1)

        threads = [
            threading.Thread(target=worker, args=(i * 10,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all observations
        assert len(tracker.acceptance_window) == min(
            tracker.window_size, num_threads * observations_per_thread
        )

    def test_percentile_accuracy(self):
        """Test that percentile calculation is accurate."""
        tracker = PercentileTracker(percentile=10.0, min_samples=10)

        # Add 1000 values from 0 to 100
        values = list(range(100))
        for val in values:
            tracker.observe(float(val))

        stats = tracker.get_stats()
        # 10th percentile of [0..99] should be around 9.9
        assert stats["percentile_threshold"] == pytest.approx(9.9, abs=1.0)

    def test_different_percentiles(self):
        """Test behavior with different percentile values."""
        values = list(range(100))

        # Test 10th percentile
        tracker_10 = PercentileTracker(percentile=10.0, min_samples=10)
        for val in values:
            tracker_10.observe(float(val))
        stats_10 = tracker_10.get_stats()

        # Test 50th percentile
        tracker_50 = PercentileTracker(percentile=50.0, min_samples=10)
        for val in values:
            tracker_50.observe(float(val))
        stats_50 = tracker_50.get_stats()

        # Test 90th percentile
        tracker_90 = PercentileTracker(percentile=90.0, min_samples=10)
        for val in values:
            tracker_90.observe(float(val))
        stats_90 = tracker_90.get_stats()

        # Thresholds should be ordered: 10th < 50th < 90th
        assert stats_10["percentile_threshold"] < stats_50["percentile_threshold"]
        assert stats_50["percentile_threshold"] < stats_90["percentile_threshold"]

    def test_fast_path_optimization(self):
        """Test that fast path is used when cache is valid."""
        tracker = PercentileTracker(percentile=50.0, min_samples=10)

        # Warmup to get past min_samples
        for i in range(20):
            tracker.observe(float(i))

        # Ensure cache is valid
        tracker.get_stats()
        assert tracker._cache_valid is True

        # Next observe_and_check should use fast path
        # We can't directly test this, but we can verify it doesn't break
        result = tracker.observe_and_check(10.0)
        assert isinstance(result, bool)
