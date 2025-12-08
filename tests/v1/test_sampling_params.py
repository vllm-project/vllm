# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SamplingParams validation, especially track_token_ids."""

import pytest

from vllm import SamplingParams


class TestTrackTokenIdsValidation:
    """Tests for track_token_ids parameter validation in SamplingParams."""

    def test_track_token_ids_none(self):
        """Test that track_token_ids=None is valid (default)."""
        params = SamplingParams(track_token_ids=None)
        assert params.track_token_ids is None

    def test_track_token_ids_empty_list(self):
        """Test that track_token_ids=[] is valid."""
        params = SamplingParams(track_token_ids=[])
        assert params.track_token_ids == []

    def test_track_token_ids_single_token(self):
        """Test that a single token ID is valid."""
        params = SamplingParams(track_token_ids=[100])
        assert params.track_token_ids == [100]

    def test_track_token_ids_multiple_tokens(self):
        """Test that multiple token IDs are valid."""
        track_ids = [100, 200, 300, 500, 1000]
        params = SamplingParams(track_token_ids=track_ids)
        assert params.track_token_ids == track_ids

    def test_track_token_ids_with_zero(self):
        """Test that token ID 0 is valid."""
        params = SamplingParams(track_token_ids=[0, 1, 2])
        assert params.track_token_ids == [0, 1, 2]

    def test_track_token_ids_large_values(self):
        """Test that large token IDs are valid (vocab bounds checked at runtime)."""
        large_ids = [100000, 150000, 200000]
        params = SamplingParams(track_token_ids=large_ids)
        assert params.track_token_ids == large_ids

    def test_track_token_ids_many_tokens(self):
        """Test that tracking many tokens (e.g., 100 for classification) is valid."""
        track_ids = list(range(100))
        params = SamplingParams(track_token_ids=track_ids)
        assert params.track_token_ids == track_ids
        assert len(params.track_token_ids) == 100

    def test_track_token_ids_negative_integer_raises(self):
        """Test that negative integers raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SamplingParams(track_token_ids=[-1])
        assert "track_token_ids must contain only non-negative integers" in str(
            exc_info.value
        )

    def test_track_token_ids_negative_in_list_raises(self):
        """Test that a negative integer in a list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SamplingParams(track_token_ids=[100, -5, 200])
        assert "track_token_ids must contain only non-negative integers" in str(
            exc_info.value
        )

    def test_track_token_ids_with_logprobs(self):
        """Test that track_token_ids works alongside logprobs parameter."""
        params = SamplingParams(
            logprobs=5,
            track_token_ids=[100, 200, 300],
        )
        assert params.logprobs == 5
        assert params.track_token_ids == [100, 200, 300]

    def test_track_token_ids_with_prompt_logprobs(self):
        """Test that track_token_ids works alongside prompt_logprobs parameter."""
        params = SamplingParams(
            prompt_logprobs=3,
            track_token_ids=[100, 200],
        )
        assert params.prompt_logprobs == 3
        assert params.track_token_ids == [100, 200]

    def test_track_token_ids_with_temperature(self):
        """Test that track_token_ids works with various temperature settings."""
        # Greedy
        params = SamplingParams(temperature=0.0, track_token_ids=[100])
        assert params.temperature == 0.0
        assert params.track_token_ids == [100]

        # Non-zero temperature
        params = SamplingParams(temperature=0.8, track_token_ids=[100])
        assert params.temperature == 0.8
        assert params.track_token_ids == [100]

    def test_track_token_ids_default_not_provided(self):
        """Test that track_token_ids is None when not provided."""
        params = SamplingParams()
        assert params.track_token_ids is None

    def test_track_token_ids_in_repr(self):
        """Test that track_token_ids appears in the string representation."""
        params = SamplingParams(track_token_ids=[100, 200])
        repr_str = repr(params)
        assert "track_token_ids" in repr_str
        assert "[100, 200]" in repr_str

    def test_track_token_ids_duplicate_values(self):
        """Test that duplicate token IDs are allowed
        (no deduplication at param level).
        """
        # Note: Deduplication happens at the batch level in merged_track_token_ids
        params = SamplingParams(track_token_ids=[100, 100, 200])
        assert params.track_token_ids == [100, 100, 200]
