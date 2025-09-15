# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for suffix cache implementation."""

import pytest

from vllm.v1.spec_decode.suffix_decode.suffix_cache import (SuffixCache,
                                                            SuffixSpecResult)


class TestSuffixCache:
    """Test suite for SuffixCache functionality."""

    def test_basic_operations(self):
        """Test basic suffix cache operations."""
        cache = SuffixCache(max_tree_depth=32, max_cached_requests=10)

        # Start a request
        cache.start_request("req1", [1, 2, 3, 4, 5])
        assert "req1" in cache.active_requests

        # Add response tokens
        cache.add_active_response("req1", [6, 7, 8])

        # Speculate based on pattern
        result = cache.speculate("req1", [1, 2, 3, 4, 5, 6])
        assert isinstance(result, SuffixSpecResult)
        assert result.token_ids == [7, 8]
        assert result.score > 0

        # Stop request
        cache.stop_request("req1")
        assert "req1" not in cache.active_requests

    def test_multiple_requests(self):
        """Test handling multiple concurrent requests."""
        cache = SuffixCache(max_tree_depth=16, max_cached_requests=5)

        # Start multiple requests with similar patterns
        cache.start_request("req1", [1, 2, 3])
        cache.start_request("req2", [1, 2, 3])
        cache.start_request("req3", [4, 5, 6])

        # Add different continuations
        cache.add_active_response("req1", [4, 5])
        cache.add_active_response("req2", [4, 6])
        cache.add_active_response("req3", [7, 8])

        # Test speculation for each request
        result1 = cache.speculate("req1", [1, 2, 3, 4])
        assert 5 in result1.token_ids

        result2 = cache.speculate("req2", [1, 2, 3, 4])
        assert 6 in result2.token_ids

        result3 = cache.speculate("req3", [4, 5, 6, 7])
        assert 8 in result3.token_ids

        # Cleanup
        cache.stop_request("req1")
        cache.stop_request("req2")
        cache.stop_request("req3")

    def test_cache_eviction(self):
        """Test cache eviction when max requests is reached."""
        cache = SuffixCache(max_tree_depth=8, max_cached_requests=3)

        # Fill cache
        for i in range(4):
            cache.start_request(f"req{i}", [i, i + 1])
            cache.add_active_response(f"req{i}", [i + 2])
            cache.stop_request(f"req{i}")

        # Check that we have at most max_cached_requests
        assert len(cache.cached_requests) <= 3

    def test_pattern_matching(self):
        """Test pattern matching with various lengths."""
        cache = SuffixCache(max_tree_depth=32)

        # Add a long sequence
        cache.start_request("req1", list(range(20)))
        cache.add_active_response("req1", list(range(20, 30)))

        # Test different pattern lengths
        # Short pattern
        result = cache.speculate("req1", [0, 1, 2])
        assert result.match_len == 3

        # Medium pattern
        result = cache.speculate("req1", list(range(10)))
        assert result.match_len == 10

        # Pattern extending into response
        result = cache.speculate("req1", list(range(25)))
        assert result.match_len == 25
        assert result.token_ids  # Should have predictions

        cache.stop_request("req1")

    def test_empty_patterns(self):
        """Test handling of empty patterns and edge cases."""
        cache = SuffixCache(max_tree_depth=16)

        # Empty prompt
        cache.start_request("req1", [])
        result = cache.speculate("req1", [])
        assert result.token_ids == []
        assert result.score == 0.0

        # Add tokens and test
        cache.add_active_response("req1", [1, 2, 3])
        result = cache.speculate("req1", [1])
        # Should predict at least the next token
        assert len(result.token_ids) >= 1
        assert result.token_ids[0] == 2

        cache.stop_request("req1")

    def test_invalid_operations(self):
        """Test error handling for invalid operations."""
        cache = SuffixCache(max_tree_depth=16)

        # Speculate on non-existent request
        with pytest.raises(ValueError, match="not active"):
            cache.speculate("nonexistent", [1, 2, 3])

        # Stop non-existent request
        with pytest.raises(ValueError, match="not active"):
            cache.stop_request("nonexistent")

        # Start duplicate request
        cache.start_request("req1", [1, 2, 3])
        with pytest.raises(ValueError, match="already active"):
            cache.start_request("req1", [4, 5, 6])

        cache.stop_request("req1")

    def test_max_depth_handling(self):
        """Test that patterns longer than max_depth are handled correctly."""
        cache = SuffixCache(max_tree_depth=8)

        # Add sequence longer than max_depth
        long_prompt = list(range(20))
        cache.start_request("req1", long_prompt)
        cache.add_active_response("req1", [100, 101, 102])

        # Pattern longer than max_depth should be truncated
        long_pattern = list(range(15)) + [100]
        result = cache.speculate("req1", long_pattern)

        # Should still find matches based on truncated pattern
        assert result.token_ids
        assert result.match_len <= 8  # Limited by max_depth

        cache.stop_request("req1")

    def test_speculation_parameters(self):
        """Test different speculation parameters."""
        cache = SuffixCache(max_tree_depth=32)

        cache.start_request("req1", [1, 2, 3])
        cache.add_active_response("req1", list(range(4, 20)))

        # Test with different max_spec_tokens
        result1 = cache.speculate("req1", [1, 2, 3, 4], max_spec_tokens=2)
        assert len(result1.token_ids) <= 2

        result2 = cache.speculate("req1", [1, 2, 3, 4], max_spec_tokens=10)
        assert len(result2.token_ids) <= 10

        # Test with different min_token_prob
        result3 = cache.speculate("req1", [1, 2, 3, 4],
                                  max_spec_tokens=5,
                                  min_token_prob=0.9)
        # With high probability threshold, might get fewer tokens
        assert len(result3.token_ids) <= 5

        cache.stop_request("req1")


if __name__ == "__main__":
    pytest.main([__file__])
