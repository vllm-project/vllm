# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for audio chunk splitting and timestamp offset tracking.

These tests verify that:
1. _split_audio_with_start_offsets() returns true split boundaries
2. _split_audio() backward compatibility (returns only chunks)
3. _find_split_point() falls back to start_idx when window too small
"""

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
    OpenAISpeechToText,
)


@pytest.fixture
def stt_instance():
    """Create a minimal OpenAISpeechToText-like object with only
    the attributes needed by _split_audio / _find_split_point."""
    obj = object.__new__(OpenAISpeechToText)
    obj.asr_config = SpeechToTextConfig(
        sample_rate=16_000,
        max_audio_clip_s=30,
        overlap_chunk_second=1,
        min_energy_split_window_size=1600,
    )
    return obj


class TestSplitAudioWithStartOffsets:
    """Tests for _split_audio_with_start_offsets()."""

    def test_short_audio_no_split(self, stt_instance):
        """Audio shorter than max_audio_clip_s should produce one chunk
        starting at offset 0.0."""
        sr = 16_000
        # 10 seconds of audio (shorter than 30s max)
        audio = np.zeros(sr * 10, dtype=np.float32)
        chunks, offsets = stt_instance._split_audio_with_start_offsets(
            audio, sr
        )
        assert len(chunks) == 1
        assert len(offsets) == 1
        assert offsets[0] == 0.0
        assert len(chunks[0]) == sr * 10

    def test_exact_chunk_boundary(self, stt_instance):
        """Audio exactly at max_audio_clip_s should produce one chunk."""
        sr = 16_000
        audio = np.zeros(sr * 30, dtype=np.float32)
        chunks, offsets = stt_instance._split_audio_with_start_offsets(
            audio, sr
        )
        assert len(chunks) == 1
        assert offsets[0] == 0.0

    def test_offsets_match_actual_split_points(self, stt_instance):
        """For audio > 30s, offsets must reflect the actual split point,
        not the nominal N * 30 boundary."""
        sr = 16_000
        duration_s = 65  # ~2.17 chunks
        total_samples = sr * duration_s

        # Create audio that is loud everywhere except for a deliberate
        # silent region near the expected split search window.
        # The split search window for chunk 0 is:
        #   [chunk_size - overlap, chunk_size] = [29s, 30s] in samples
        # Place silence at 29.5s so the split happens there, not at 30s.
        audio = np.ones(total_samples, dtype=np.float32)
        silence_center = int(29.5 * sr)
        silence_half = 800  # half a min_energy_window
        audio[silence_center - silence_half : silence_center + silence_half] = (
            0.0
        )

        chunks, offsets = stt_instance._split_audio_with_start_offsets(
            audio, sr
        )

        # Should have at least 2 chunks
        assert len(chunks) >= 2
        assert len(offsets) == len(chunks)

        # First chunk always starts at 0
        assert offsets[0] == 0.0

        # Second chunk should start near 29.5s, NOT at 30.0s
        # The actual split point depends on the energy search, but it should
        # be within the overlap window [29s, 30s] and closer to the silence.
        assert offsets[1] != 30.0, (
            "Offset should reflect actual split point, not nominal 30s"
        )
        assert 29.0 <= offsets[1] <= 30.0

        # Verify chunks cover all samples
        total_chunk_samples = sum(len(c) for c in chunks)
        assert total_chunk_samples == total_samples

    def test_offsets_are_monotonically_increasing(self, stt_instance):
        """Chunk start offsets must be strictly increasing."""
        sr = 16_000
        audio = np.random.RandomState(42).randn(sr * 95).astype(np.float32)
        chunks, offsets = stt_instance._split_audio_with_start_offsets(
            audio, sr
        )
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1]

    def test_chunk_count_matches_offset_count(self, stt_instance):
        """Number of offsets must equal number of chunks."""
        sr = 16_000
        audio = np.random.RandomState(0).randn(sr * 120).astype(np.float32)
        chunks, offsets = stt_instance._split_audio_with_start_offsets(
            audio, sr
        )
        assert len(chunks) == len(offsets)


class TestSplitAudioBackwardCompat:
    """Tests for _split_audio() backward compatibility."""

    def test_returns_only_chunks(self, stt_instance):
        """_split_audio() should return a plain list of arrays (no offsets)."""
        sr = 16_000
        audio = np.random.RandomState(1).randn(sr * 65).astype(np.float32)
        result = stt_instance._split_audio(audio, sr)
        assert isinstance(result, list)
        assert all(isinstance(c, np.ndarray) for c in result)

    def test_same_chunks_as_with_offsets(self, stt_instance):
        """_split_audio() chunks should be identical to
        _split_audio_with_start_offsets() chunks."""
        sr = 16_000
        audio = np.random.RandomState(2).randn(sr * 65).astype(np.float32)
        chunks_only = stt_instance._split_audio(audio, sr)
        chunks_with, _ = stt_instance._split_audio_with_start_offsets(
            audio, sr
        )
        assert len(chunks_only) == len(chunks_with)
        for a, b in zip(chunks_only, chunks_with):
            np.testing.assert_array_equal(a, b)


class TestFindSplitPoint:
    """Tests for _find_split_point()."""

    def test_finds_quietest_region(self, stt_instance):
        """Should return the index of the quietest window."""
        sr = 16_000
        # Create 1s of loud audio with a silent notch at 0.5s
        audio = np.ones(sr, dtype=np.float32)
        notch_center = sr // 2
        notch_half = 800
        audio[notch_center - notch_half : notch_center + notch_half] = 0.0

        split = stt_instance._find_split_point(audio, 0, sr)
        # Split should be near the silent notch
        assert abs(split - notch_center) < 1600  # within one window

    def test_fallback_to_start_idx_when_window_too_small(self, stt_instance):
        """When the search region is smaller than min_energy_window,
        the loop doesn't execute and we should get start_idx back
        (not 0)."""
        sr = 16_000
        audio = np.ones(sr * 60, dtype=np.float32)
        # Search region smaller than min_energy_window (1600 samples)
        start_idx = 50_000
        end_idx = start_idx + 100  # only 100 samples, < 1600
        split = stt_instance._find_split_point(audio, start_idx, end_idx)
        assert split == start_idx, (
            f"Expected fallback to start_idx={start_idx}, got {split}"
        )

    def test_returns_within_search_region(self, stt_instance):
        """Split point must be within [start_idx, end_idx)."""
        sr = 16_000
        audio = np.random.RandomState(3).randn(sr * 60).astype(np.float32)
        start_idx = sr * 29
        end_idx = sr * 30
        split = stt_instance._find_split_point(audio, start_idx, end_idx)
        assert start_idx <= split < end_idx
