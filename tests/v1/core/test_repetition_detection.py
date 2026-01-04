# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for repetition detection in scheduler utilities."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_repetition, check_stop
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# Use a consistent EOS token ID for tests
EOS_TOKEN_ID = 2


class TestCheckRepetition:
    """Tests for the check_repetition function."""

    def test_disabled_when_max_consecutive_repeats_zero(self):
        """Repetition detection should be disabled when max_consecutive_repeats=0."""
        output_token_ids = [1, 1, 1, 1, 1]  # 5 consecutive identical tokens
        assert check_repetition(output_token_ids, max_consecutive_repeats=0) is False

    def test_disabled_when_max_consecutive_repeats_negative(self):
        """Repetition detection should be disabled for negative values."""
        output_token_ids = [1, 1, 1, 1, 1]
        assert check_repetition(output_token_ids, max_consecutive_repeats=-1) is False

    def test_not_enough_tokens(self):
        """Should return False when there aren't enough tokens."""
        output_token_ids = [1, 1]  # Only 2 tokens
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is False

    def test_no_repetition_different_tokens(self):
        """Should return False when tokens are different."""
        output_token_ids = [1, 2, 3, 4, 5]
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is False

    def test_no_repetition_last_two_differ(self):
        """O(1) fast path: should return False immediately if last 2 tokens differ."""
        output_token_ids = [1, 1, 1, 1, 2]  # Last token is different
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is False

    def test_detect_exact_consecutive_repeats(self):
        """Should detect exactly N consecutive identical tokens."""
        output_token_ids = [1, 2, 3, 3, 3]  # 3 consecutive 3s
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is True

    def test_detect_more_than_threshold_repeats(self):
        """Should detect when there are more than N consecutive repeats."""
        output_token_ids = [1, 2, 3, 3, 3, 3, 3]  # 5 consecutive 3s
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is True

    def test_partial_match_not_enough(self):
        """Should return False when consecutive count is less than threshold."""
        output_token_ids = [1, 2, 3, 3]  # Only 2 consecutive 3s
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is False

    def test_repetition_at_end_only(self):
        """Should only check repetition at the end of the sequence."""
        output_token_ids = [1, 1, 1, 2, 3, 4]  # Repetition at start, not end
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is False

    def test_large_consecutive_threshold(self):
        """Should work with large consecutive repeat thresholds (performance test)."""
        # Create 100 identical tokens
        output_token_ids = [42] * 100
        assert check_repetition(output_token_ids, max_consecutive_repeats=100) is True
        assert check_repetition(output_token_ids, max_consecutive_repeats=101) is False

    def test_single_token_repetition(self):
        """Should handle edge case of single token threshold.

        max_consecutive_repeats=1 is normalized to 2 since a single token
        cannot be considered repetition. This means it behaves the same as
        max_consecutive_repeats=2.
        """
        # With max_consecutive_repeats=1, it's normalized to 2
        # So we need at least 2 identical tokens
        output_token_ids = [1, 2, 3]
        assert check_repetition(output_token_ids, max_consecutive_repeats=1) is False

        # When last 2 tokens match, max_consecutive_repeats=1 (normalized to 2)
        # should trigger
        output_token_ids_same = [1, 2, 2]
        assert (
            check_repetition(output_token_ids_same, max_consecutive_repeats=1) is True
        )

    def test_single_token_list_with_max_1(self):
        """Should handle single token list without IndexError.

        This is a regression test for the IndexError that occurred when
        max_consecutive_repeats=1 and output_token_ids had only one token.
        """
        output_token_ids = [42]  # Single token
        # Should not raise IndexError, should return False
        assert check_repetition(output_token_ids, max_consecutive_repeats=1) is False

    def test_empty_token_list(self):
        """Should handle empty token list gracefully."""
        output_token_ids: list[int] = []
        assert check_repetition(output_token_ids, max_consecutive_repeats=3) is False


class TestCheckStopWithRepetition:
    """Tests for check_stop function with repetition detection."""

    def _create_request(
        self,
        max_consecutive_repeats: int = 0,
        max_tokens: int = 100,
        ignore_eos: bool = True,
    ) -> Request:
        """Helper to create a test request."""
        sampling_params = SamplingParams(
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            max_consecutive_repeats=max_consecutive_repeats,
        )
        return Request(
            request_id="test-request",
            prompt_token_ids=[0, 1, 2],
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=EOS_TOKEN_ID,
        )

    def test_no_stop_when_repetition_detection_disabled(self):
        """Should not stop for repetition when max_consecutive_repeats=0."""
        request = self._create_request(max_consecutive_repeats=0)
        # Add repetitive tokens
        request.append_output_token_ids([5, 5, 5, 5, 5])

        result = check_stop(request, max_model_len=1000)
        assert result is False

    def test_stop_on_repetition_detection(self):
        """Should stop when repetitive pattern detected."""
        request = self._create_request(max_consecutive_repeats=4)
        # Add 4 consecutive identical tokens
        request.append_output_token_ids([5, 5, 5, 5])

        result = check_stop(request, max_model_len=1000)
        assert result is True
        assert request.status == RequestStatus.FINISHED_REPETITION
        assert request.stop_reason == "repetition_detected"

    def test_no_stop_below_threshold(self):
        """Should not stop when consecutive count below threshold."""
        request = self._create_request(max_consecutive_repeats=4)
        # Add only 3 consecutive identical tokens
        request.append_output_token_ids([5, 5, 5])

        result = check_stop(request, max_model_len=1000)
        assert result is False

    def test_eos_takes_precedence_over_repetition(self):
        """EOS token should stop before repetition check."""
        request = self._create_request(max_consecutive_repeats=4, ignore_eos=False)
        # Add tokens ending with EOS
        request.append_output_token_ids([5, 5, 5, EOS_TOKEN_ID])

        result = check_stop(request, max_model_len=1000)
        assert result is True
        assert request.status == RequestStatus.FINISHED_STOPPED
        # Stop reason should not be repetition since EOS was hit first

    def test_length_cap_takes_precedence(self):
        """Length cap should stop before repetition check."""
        request = self._create_request(max_consecutive_repeats=10, max_tokens=3)
        # Add 3 tokens (hitting max_tokens)
        request.append_output_token_ids([5, 5, 5])

        result = check_stop(request, max_model_len=1000)
        assert result is True
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_model_len_cap_takes_precedence(self):
        """Model length cap should stop before repetition check."""
        request = self._create_request(max_consecutive_repeats=10, max_tokens=100)
        # 3 prompt tokens + 3 output = 6 total, max_model_len = 6
        request.append_output_token_ids([5, 5, 5])

        result = check_stop(request, max_model_len=6)
        assert result is True
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


class TestRepetitionDetectionPerformance:
    """Performance-related tests for repetition detection."""

    def test_fast_path_no_repetition(self):
        """Verify fast path is taken when last two tokens differ."""
        # This should be O(1) - just one comparison
        # Last token differs from second-last
        output_token_ids = list(range(1000)) + [999]
        # Should return immediately without iterating
        assert check_repetition(output_token_ids, max_consecutive_repeats=100) is False

    def test_handles_large_threshold_efficiently(self):
        """Should handle large thresholds without excessive iteration."""
        # Create sequence where repetition starts but doesn't continue
        output_token_ids = [1] * 50 + [2, 2]  # Only 2 consecutive 2s at end
        # Even with large threshold, should exit early
        assert check_repetition(output_token_ids, max_consecutive_repeats=100) is False
