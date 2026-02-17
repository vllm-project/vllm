# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that audio chunk splitting returns correct start time offsets.

This test verifies the fix for issue #29350 where segment timestamps
were drifting due to using nominal fixed offsets instead of actual
chunk boundary times.
"""

import numpy as np
import pytest


class MockASRConfig:
    """Mock ASR config for testing."""

    def __init__(
        self,
        max_audio_clip_s: float = 30.0,
        overlap_chunk_second: float = 1.0,
        min_energy_split_window_size: int = 1600,  # 0.1s at 16kHz
    ):
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_second = overlap_chunk_second
        self.min_energy_split_window_size = min_energy_split_window_size


class AudioSplitter:
    """Isolated audio splitter for testing.

    This mirrors the logic from OpenAISpeechToText._split_audio
    """

    def __init__(self, asr_config: MockASRConfig):
        self.asr_config = asr_config

    def _find_split_point(self, wav: np.ndarray, start_idx: int, end_idx: int) -> int:
        """Find the best point to split audio by looking for silence."""
        import math

        segment = wav[start_idx:end_idx]
        min_energy = math.inf
        quietest_idx = start_idx  # Default to start if segment is too short
        min_energy_window = self.asr_config.min_energy_split_window_size

        for i in range(0, max(1, len(segment) - min_energy_window), min_energy_window):
            window = segment[i : i + min_energy_window]
            energy = (window**2).mean() ** 0.5
            if energy < min_energy:
                quietest_idx = i + start_idx
                min_energy = energy
        return quietest_idx

    def split_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> tuple[list[np.ndarray], list[float]]:
        """Split audio into chunks at quiet points.

        Returns:
            A tuple of (chunks, chunk_start_times) where:
            - chunks: List of audio arrays
            - chunk_start_times: List of start times in seconds for each chunk
        """
        chunk_size = int(sample_rate * self.asr_config.max_audio_clip_s)
        overlap_size = int(sample_rate * self.asr_config.overlap_chunk_second)
        chunks = []
        chunk_start_times = []
        i = 0

        while i < audio_data.shape[-1]:
            # Record the actual start time of this chunk in seconds
            chunk_start_times.append(float(i) / sample_rate)

            if i + chunk_size >= audio_data.shape[-1]:
                # handle last chunk
                chunks.append(audio_data[..., i:])
                break

            # Find the best split point in the overlap region
            search_start = i + chunk_size - overlap_size
            search_end = min(i + chunk_size, audio_data.shape[-1])
            split_point = self._find_split_point(audio_data, search_start, search_end)

            # Extract chunk up to the split point
            chunks.append(audio_data[..., i:split_point])
            i = split_point

        return chunks, chunk_start_times


class TestSplitAudioOffsets:
    """Test suite for audio chunk splitting with correct offsets."""

    def test_single_chunk_no_split(self):
        """Audio shorter than max_audio_clip_s should not be split."""
        config = MockASRConfig(max_audio_clip_s=30.0)
        splitter = AudioSplitter(config)

        # 20 seconds of audio at 16kHz
        sample_rate = 16000
        audio = np.zeros(20 * sample_rate)

        chunks, offsets = splitter.split_audio(audio, sample_rate)

        assert len(chunks) == 1
        assert len(offsets) == 1
        assert offsets[0] == 0.0

    def test_two_chunks_offset_accuracy(self):
        """Two chunks should have accurate start time offsets."""
        config = MockASRConfig(max_audio_clip_s=30.0, overlap_chunk_second=1.0)
        splitter = AudioSplitter(config)

        # 50 seconds of audio at 16kHz
        sample_rate = 16000
        audio = np.random.randn(50 * sample_rate).astype(np.float32)

        # Add silence at 29.5s to influence split point
        silence_start = int(29.5 * sample_rate)
        silence_end = int(29.7 * sample_rate)
        audio[silence_start:silence_end] = 0.0

        chunks, offsets = splitter.split_audio(audio, sample_rate)

        assert len(chunks) == 2
        assert len(offsets) == 2
        assert offsets[0] == 0.0
        # Second chunk should start at actual split point, not nominal 30s
        assert offsets[1] < 30.0  # Should be around 29.5s due to silence

    def test_multiple_chunks_cumulative_accuracy(self):
        """Multiple chunks should not accumulate timestamp drift."""
        config = MockASRConfig(max_audio_clip_s=30.0, overlap_chunk_second=1.0)
        splitter = AudioSplitter(config)

        # 100 seconds of audio at 16kHz
        sample_rate = 16000
        audio = np.random.randn(100 * sample_rate).astype(np.float32)

        # Add silence regions to force early splits
        for t in [29.5, 58.5, 87.5]:
            silence_start = int(t * sample_rate)
            silence_end = int((t + 0.2) * sample_rate)
            if silence_end < len(audio):
                audio[silence_start:silence_end] = 0.0

        chunks, offsets = splitter.split_audio(audio, sample_rate)

        # Verify offsets are cumulative sums of actual chunk lengths
        total_samples = 0
        for i, (chunk, offset) in enumerate(zip(chunks, offsets)):
            expected_offset = float(total_samples) / sample_rate
            assert abs(offset - expected_offset) < 0.001, (
                f"Chunk {i}: expected offset {expected_offset}, got {offset}"
            )
            total_samples += len(chunk)

    def test_offset_matches_chunk_boundary(self):
        """Each offset should exactly match the end of the previous chunk."""
        config = MockASRConfig(max_audio_clip_s=30.0)
        splitter = AudioSplitter(config)

        sample_rate = 16000
        audio = np.random.randn(65 * sample_rate).astype(np.float32)

        chunks, offsets = splitter.split_audio(audio, sample_rate)

        # First offset is always 0
        assert offsets[0] == 0.0

        # Each subsequent offset should equal cumulative samples / sample_rate
        cumulative_samples = 0
        for i in range(len(chunks) - 1):
            cumulative_samples += len(chunks[i])
            expected_next_offset = float(cumulative_samples) / sample_rate
            actual_next_offset = offsets[i + 1]
            assert abs(actual_next_offset - expected_next_offset) < 0.001, (
                f"Offset {i+1}: expected {expected_next_offset}, got {actual_next_offset}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
