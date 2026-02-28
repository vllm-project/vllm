# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for video frame validation logic used in Qwen2VL/Qwen3VL multimodal
processing. These tests verify the temporal_patch_size validation algorithm
without requiring model downloads.
"""

import pytest

pytestmark = pytest.mark.cpu_test


def validate_video_frames_for_temporal_patch_size(
    num_frames: int,
    temporal_patch_size: int,
) -> None:
    """
    Validate that video frame count is a multiple of temporal_patch_size.

    This is a standalone implementation of the validation logic used in
    Qwen2VL/Qwen3VL multimodal processing.

    Args:
        num_frames: Number of video frames
        temporal_patch_size: The temporal patch size (typically 2 for Qwen models)

    Raises:
        ValueError: If num_frames is not a multiple of temporal_patch_size
    """
    if temporal_patch_size > 1 and num_frames % temporal_patch_size != 0:
        raise ValueError(
            f"Video frame count ({num_frames}) must be a multiple "
            f"of temporal_patch_size ({temporal_patch_size}). "
            f"When using pre-processed video (do_sample_frames=False), "
            f"ensure the frame count is properly aligned. "
            f"Consider using qwen-vl-utils smart_nframes() or "
            f"round_by_factor() to sample the correct number of frames."
        )


class TestVideoFrameValidation:
    """Tests for video frame validation with temporal_patch_size alignment."""

    @pytest.mark.parametrize(
        ("num_frames", "temporal_patch_size"),
        [
            # Valid cases for temporal_patch_size=2
            (2, 2),
            (4, 2),
            (6, 2),
            (8, 2),
            (10, 2),
            (100, 2),
            # Valid cases for temporal_patch_size=1 (any count is valid)
            (1, 1),
            (3, 1),
            (5, 1),
            (7, 1),
            # Valid cases for temporal_patch_size=4
            (4, 4),
            (8, 4),
            (12, 4),
        ],
    )
    def test_valid_frame_counts_pass(
        self,
        num_frames: int,
        temporal_patch_size: int,
    ) -> None:
        """Test that valid frame counts (multiples of temporal_patch_size) pass."""
        # Should not raise
        validate_video_frames_for_temporal_patch_size(num_frames, temporal_patch_size)

    @pytest.mark.parametrize(
        ("num_frames", "temporal_patch_size"),
        [
            # Invalid cases for temporal_patch_size=2
            (1, 2),
            (3, 2),
            (5, 2),
            (7, 2),
            (9, 2),
            (11, 2),
            (99, 2),
            # Invalid cases for temporal_patch_size=4
            (1, 4),
            (2, 4),
            (3, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (9, 4),
        ],
    )
    def test_invalid_frame_counts_raise_error(
        self,
        num_frames: int,
        temporal_patch_size: int,
    ) -> None:
        """Test that invalid frame counts raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            validate_video_frames_for_temporal_patch_size(
                num_frames, temporal_patch_size
            )

        error_msg = str(exc_info.value)
        assert str(num_frames) in error_msg
        assert str(temporal_patch_size) in error_msg
        assert "temporal_patch_size" in error_msg
        assert "qwen-vl-utils" in error_msg or "smart_nframes" in error_msg

    def test_error_message_is_informative(self) -> None:
        """Test that the error message provides actionable guidance."""
        with pytest.raises(ValueError) as exc_info:
            validate_video_frames_for_temporal_patch_size(5, 2)

        error_msg = str(exc_info.value)

        # Check that key information is present
        assert "5" in error_msg  # num_frames
        assert "2" in error_msg  # temporal_patch_size
        assert "do_sample_frames" in error_msg  # Context about pre-processed video
        assert "smart_nframes" in error_msg or "round_by_factor" in error_msg

    def test_temporal_patch_size_one_always_valid(self) -> None:
        """Test that temporal_patch_size=1 accepts any frame count."""
        for num_frames in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 100, 999]:
            # Should not raise for any frame count
            validate_video_frames_for_temporal_patch_size(num_frames, 1)

    def test_single_frame_with_patch_size_two_fails(self) -> None:
        """Test that single-frame video fails with temporal_patch_size=2."""
        with pytest.raises(ValueError):
            validate_video_frames_for_temporal_patch_size(1, 2)

    def test_large_odd_frame_count_fails(self) -> None:
        """Test that large odd frame counts still fail validation."""
        with pytest.raises(ValueError):
            validate_video_frames_for_temporal_patch_size(1001, 2)

    def test_large_even_frame_count_passes(self) -> None:
        """Test that large even frame counts pass validation."""
        # Should not raise
        validate_video_frames_for_temporal_patch_size(1000, 2)


class TestQwenFrameFactorCompatibility:
    """
    Tests verifying compatibility with qwen-vl-utils FRAME_FACTOR.

    qwen-vl-utils uses FRAME_FACTOR=2 which matches temporal_patch_size=2.
    The round_by_factor, ceil_by_factor, and floor_by_factor functions
    ensure frame counts are always multiples of FRAME_FACTOR.
    """

    FRAME_FACTOR = 2  # From qwen-vl-utils

    @staticmethod
    def round_by_factor(number: int, factor: int) -> int:
        """Round number to nearest multiple of factor (from qwen-vl-utils)."""
        return round(number / factor) * factor

    @staticmethod
    def floor_by_factor(number: int, factor: int) -> int:
        """Round down to nearest multiple of factor (from qwen-vl-utils)."""
        return (number // factor) * factor

    @staticmethod
    def ceil_by_factor(number: int, factor: int) -> int:
        """Round up to nearest multiple of factor (from qwen-vl-utils)."""
        return ((number + factor - 1) // factor) * factor

    @pytest.mark.parametrize("raw_frames", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    def test_round_by_factor_produces_valid_counts(self, raw_frames: int) -> None:
        """Test that round_by_factor output passes validation."""
        adjusted = self.round_by_factor(raw_frames, self.FRAME_FACTOR)
        if adjusted >= self.FRAME_FACTOR:  # Ensure at least FRAME_FACTOR frames
            validate_video_frames_for_temporal_patch_size(adjusted, self.FRAME_FACTOR)

    @pytest.mark.parametrize("raw_frames", [2, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    def test_floor_by_factor_produces_valid_counts(self, raw_frames: int) -> None:
        """Test that floor_by_factor output passes validation."""
        adjusted = self.floor_by_factor(raw_frames, self.FRAME_FACTOR)
        if adjusted >= self.FRAME_FACTOR:  # Ensure at least FRAME_FACTOR frames
            validate_video_frames_for_temporal_patch_size(adjusted, self.FRAME_FACTOR)

    @pytest.mark.parametrize("raw_frames", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    def test_ceil_by_factor_produces_valid_counts(self, raw_frames: int) -> None:
        """Test that ceil_by_factor output passes validation."""
        adjusted = self.ceil_by_factor(raw_frames, self.FRAME_FACTOR)
        validate_video_frames_for_temporal_patch_size(adjusted, self.FRAME_FACTOR)
