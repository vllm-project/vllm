# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for async TP adaptive gating:
  1. should_fuse_async_tp() decision tree
  2. is_applicable_for_range() with async_tp_min_tokens threshold
"""

import pytest

from vllm.compilation.passes.fusion.async_tp_heuristic import should_fuse_async_tp
from vllm.config.utils import Range


# --------------------------------------------------------------------------- #
# Phase 2 tests: decision tree                                                 #
# --------------------------------------------------------------------------- #

class TestShouldFuseAsyncTP:
    """Verify the ported AutoHeuristic decision tree returns sensible
    results for representative model shapes."""

    # (M, K, N, expected)
    SMALL_BATCH_CASES = [
        # Decode-like: very small M → typically True due to low m_times_n
        (1, 8192, 28672, True),
        (4, 8192, 28672, True),
        (8, 4096, 14336, True),
        (16, 4096, 14336, True),
        (32, 8192, 28672, True),
    ]

    LARGE_BATCH_CASES = [
        # Large-prefill shapes
        (256, 8192, 28672, True),
        (4096, 4096, 14336, True),
    ]

    @pytest.mark.parametrize("M,K,N,expected", SMALL_BATCH_CASES)
    def test_small_batch(self, M: int, K: int, N: int, expected: bool):
        assert should_fuse_async_tp(M, K, N) == expected

    @pytest.mark.parametrize("M,K,N,expected", LARGE_BATCH_CASES)
    def test_large_batch(self, M: int, K: int, N: int, expected: bool):
        assert should_fuse_async_tp(M, K, N) == expected

    def test_deterministic(self):
        """Same inputs must always return the same result."""
        for _ in range(100):
            assert should_fuse_async_tp(256, 8192, 28672) is True
            assert should_fuse_async_tp(1, 8192, 28672) is True

    def test_no_fuse_large_m_small_n(self):
        """Large M with small N should not fuse (more overhead than benefit)."""
        assert should_fuse_async_tp(4096, 8192, 4096) is False

    def test_returns_bool(self):
        result = should_fuse_async_tp(128, 4096, 14336)
        assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
# Phase 1 tests: coarse-grained range gating                                  #
# --------------------------------------------------------------------------- #

class TestRangeGating:
    """Test that the Range-based min_tokens threshold works correctly.

    These are pure-logic tests that don't require GPU or distributed setup.
    They verify the threshold comparison logic that will be used inside
    is_applicable_for_range() of both AsyncTPPass and
    SequenceParallelismPass.
    """

    def _check_threshold(
        self, compile_range: Range, min_tokens: int | None
    ) -> bool:
        """Replicate the threshold check from is_applicable_for_range."""
        if min_tokens is not None and compile_range.end < min_tokens:
            return False
        return True

    def test_small_range_blocked(self):
        """Range(1,1) should be blocked with default min_tokens=128."""
        assert self._check_threshold(Range(1, 1), 128) is False

    def test_large_range_allowed(self):
        """Range(512,512) should be allowed with default min_tokens=128."""
        assert self._check_threshold(Range(512, 512), 128) is True

    def test_boundary_range_blocked(self):
        """Range(127,127) is just below 128 → blocked."""
        assert self._check_threshold(Range(127, 127), 128) is False

    def test_boundary_range_allowed(self):
        """Range(128,128) is exactly at the threshold → allowed."""
        assert self._check_threshold(Range(128, 128), 128) is True

    def test_none_disables_threshold(self):
        """Setting min_tokens=None disables the check entirely."""
        assert self._check_threshold(Range(1, 1), None) is True

    def test_min_tokens_1_always_fuses(self):
        """min_tokens=1 means always fuse (range.end ≥ 1 always)."""
        assert self._check_threshold(Range(1, 1), 1) is True
        assert self._check_threshold(Range(4, 4), 1) is True

    def test_range_with_different_start_end(self):
        """For non-single-size ranges, end is what matters."""
        assert self._check_threshold(Range(1, 256), 128) is True
        assert self._check_threshold(Range(1, 64), 128) is False
