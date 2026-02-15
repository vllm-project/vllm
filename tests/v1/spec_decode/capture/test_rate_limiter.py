# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for rate limiting in distillation capture."""

import threading

import pytest

from vllm.v1.spec_decode.capture.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test RateLimiter functionality."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_percentage=10.0)
        assert limiter.max_percentage == 10.0
        assert limiter.total_drafts == 0
        assert limiter.captured_drafts == 0

    def test_first_draft_always_allowed(self):
        """Test that first draft is always allowed to be captured."""
        limiter = RateLimiter(max_percentage=10.0)
        assert limiter.should_capture() is True
        assert limiter.total_drafts == 1

    def test_should_capture_increments_total(self):
        """Test that should_capture increments total_drafts counter."""
        limiter = RateLimiter(max_percentage=50.0)
        initial_total = limiter.total_drafts

        limiter.should_capture()
        assert limiter.total_drafts == initial_total + 1

        limiter.should_capture()
        assert limiter.total_drafts == initial_total + 2

    def test_record_captured_increments_counter(self):
        """Test that record_captured increments captured_drafts counter."""
        limiter = RateLimiter(max_percentage=50.0)
        initial_captured = limiter.captured_drafts

        limiter.record_captured()
        assert limiter.captured_drafts == initial_captured + 1

        limiter.record_captured()
        assert limiter.captured_drafts == initial_captured + 2

    def test_rate_limiting_enforced(self):
        """Test that rate limiting is enforced correctly."""
        limiter = RateLimiter(max_percentage=10.0)

        # Capture first draft (always allowed)
        assert limiter.should_capture() is True
        limiter.record_captured()

        # Now we have 1/1 = 100% captured
        # Next 9 drafts should be rejected to bring percentage down
        for _ in range(9):
            assert limiter.should_capture() is False

        # Now we have 1/10 = 10% captured
        # Next draft should be allowed
        assert limiter.should_capture() is True

    def test_rate_limiting_50_percent(self):
        """Test rate limiting at 50%."""
        limiter = RateLimiter(max_percentage=50.0)

        # Capture first draft
        assert limiter.should_capture() is True
        limiter.record_captured()

        # Should reject next draft (would be 2/2 = 100%)
        assert limiter.should_capture() is False

        # Now 1/2 = 50%, should allow next
        assert limiter.should_capture() is True
        limiter.record_captured()

        # Now 2/3 = 66.7%, should reject
        assert limiter.should_capture() is False

    def test_get_stats_initial(self):
        """Test get_stats with no drafts processed."""
        limiter = RateLimiter(max_percentage=10.0)
        stats = limiter.get_stats()

        assert stats["total_drafts"] == 0
        assert stats["captured_drafts"] == 0
        assert stats["percentage"] == 0.0

    def test_get_stats_after_capturing(self):
        """Test get_stats after processing some drafts."""
        limiter = RateLimiter(max_percentage=25.0)

        for _ in range(10):
            if limiter.should_capture():
                limiter.record_captured()

        stats = limiter.get_stats()
        assert stats["total_drafts"] == 10
        assert stats["captured_drafts"] == 3
        assert stats["percentage"] == 30.0

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = RateLimiter(max_percentage=50.0)
        num_threads = 10
        calls_per_thread = 100

        def worker():
            for _ in range(calls_per_thread):
                if limiter.should_capture():
                    limiter.record_captured()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = limiter.get_stats()
        # Total should be exactly num_threads * calls_per_thread
        assert stats["total_drafts"] == num_threads * calls_per_thread
        # Captured should be approximately 50% (within reasonable margin)
        expected_captured = (num_threads * calls_per_thread) * 0.5
        assert abs(stats["captured_drafts"] - expected_captured) < expected_captured * 0.2

    def test_zero_percent_limit(self):
        """Test behavior with 0% limit (should only capture first draft)."""
        limiter = RateLimiter(max_percentage=0.0)

        # First draft always allowed
        assert limiter.should_capture() is True
        limiter.record_captured()

        # All subsequent drafts should be rejected
        for _ in range(100):
            assert limiter.should_capture() is False

        stats = limiter.get_stats()
        assert stats["captured_drafts"] == 1
        assert stats["total_drafts"] == 101

    def test_hundred_percent_limit(self):
        """Test behavior with 100% limit (should capture all drafts)."""
        limiter = RateLimiter(max_percentage=100.0)

        # All drafts should be allowed
        for _ in range(100):
            assert limiter.should_capture() is True
            limiter.record_captured()

        stats = limiter.get_stats()
        assert stats["captured_drafts"] == 100
        assert stats["total_drafts"] == 100
        assert stats["percentage"] == 100.0

    def test_backward_compat_should_log(self):
        """Test backward compatibility alias should_log."""
        limiter = RateLimiter(max_percentage=100.0)
        # should_log should work as alias for should_capture
        assert limiter.should_log() is True
        limiter.record_logged()
        assert limiter.captured_drafts == 1

    def test_backward_compat_record_logged(self):
        """Test backward compatibility alias record_logged."""
        limiter = RateLimiter(max_percentage=100.0)
        limiter.should_capture()
        limiter.record_logged()  # Should work as alias
        assert limiter.captured_drafts == 1
