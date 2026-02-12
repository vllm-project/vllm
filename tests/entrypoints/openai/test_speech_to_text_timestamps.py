# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for audio chunking and timestamp accuracy.

These tests verify that segment-level timestamps remain accurate when
long audio is split into variable-length chunks.  Prior to the fix for
https://github.com/vllm-project/vllm/issues/32588, the code assumed
every chunk had the same nominal duration (max_audio_clip_s), causing
cumulative timestamp drift when _split_audio clips at silence points.

The _split_audio and _find_split_point logic is copied here verbatim
from vllm/entrypoints/openai/speech_to_text/speech_to_text.py so that
the tests can run without heavy GPU-only dependencies (triton, CUDA,
mistral_common>=1.9.1 which requires Python<3.14, etc.).
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Standalone copies of split logic from speech_to_text.py
# (avoids importing the full vLLM dependency tree)
# ---------------------------------------------------------------------------
@dataclass
class _MockConfig:
    """Minimal mirror of SpeechToTextConfig for testing."""

    sample_rate: float = 16_000
    max_audio_clip_s: int = 30
    overlap_chunk_second: int = 1
    min_energy_split_window_size: int = 1600


def _find_split_point(
    wav: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window: int,
) -> int:
    """Verbatim copy of OpenAISpeechToText._find_split_point."""
    segment = wav[start_idx:end_idx]
    min_energy = math.inf
    quietest_idx = 0
    for i in range(0, len(segment) - min_energy_window, min_energy_window):
        window = segment[i : i + min_energy_window]
        energy = (window**2).mean() ** 0.5
        if energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy
    return quietest_idx


def _split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    cfg: _MockConfig,
) -> list[np.ndarray]:
    """Verbatim copy of OpenAISpeechToText._split_audio."""
    assert cfg.max_audio_clip_s is not None
    chunk_size = sample_rate * cfg.max_audio_clip_s
    overlap_size = sample_rate * cfg.overlap_chunk_second
    chunks: list[np.ndarray] = []
    i = 0
    while i < audio_data.shape[-1]:
        if i + chunk_size >= audio_data.shape[-1]:
            chunks.append(audio_data[..., i:])
            break
        search_start = i + chunk_size - overlap_size
        search_end = min(i + chunk_size, audio_data.shape[-1])
        split_point = _find_split_point(
            audio_data,
            search_start,
            search_end,
            cfg.min_energy_split_window_size,
        )
        chunks.append(audio_data[..., i:split_point])
        i = split_point
    return chunks


# ---------------------------------------------------------------------------
# Timestamp accumulation tests  (pure-Python, no vllm dependency)
# ---------------------------------------------------------------------------
class TestTimestampAccumulation:
    """Verify that using accumulated chunk durations produces correct
    start_time values, unlike the old idx * chunk_size approach."""

    @staticmethod
    def _old_buggy_start_time(idx: int, chunk_size_s: float) -> float:
        """The old (buggy) calculation."""
        return float(idx * chunk_size_s)

    @staticmethod
    def _new_accumulated_start_time(
        chunk_durations: list[float], chunk_idx: int
    ) -> float:
        """The new (correct) calculation: sum of all previous durations."""
        return sum(chunk_durations[:chunk_idx])

    def test_no_drift_with_accumulated_durations(self):
        """Accumulated durations should produce zero drift."""
        chunk_durations = [29.5, 29.8, 29.2, 28.9, 29.6]

        expected_starts = [0.0]
        for d in chunk_durations[:-1]:
            expected_starts.append(expected_starts[-1] + d)

        for idx in range(len(chunk_durations)):
            calculated = self._new_accumulated_start_time(chunk_durations, idx)
            assert math.isclose(calculated, expected_starts[idx], abs_tol=1e-9), (
                f"Chunk {idx}: expected {expected_starts[idx]}, got {calculated}"
            )

    def test_old_logic_has_drift(self):
        """The old logic DOES produce drift when chunks are variable."""
        chunk_size_s = 30.0
        chunk_durations = [29.5, 29.8, 29.2]

        actual_start = 0.0
        for idx, dur in enumerate(chunk_durations):
            old_start = self._old_buggy_start_time(idx, chunk_size_s)
            if idx > 0:
                assert not math.isclose(old_start, actual_start, abs_tol=0.01), (
                    f"Chunk {idx}: old logic should have drift but "
                    f"old_start={old_start}, actual_start={actual_start}"
                )
            actual_start += dur

    def test_single_chunk_no_difference(self):
        """With a single chunk, both old and new logic should agree."""
        chunk_durations = [29.5]
        old_start = self._old_buggy_start_time(0, 30.0)
        new_start = self._new_accumulated_start_time(chunk_durations, 0)
        assert old_start == new_start == 0.0

    def test_equal_chunks_no_difference(self):
        """When all chunks are exactly chunk_size, old and new agree."""
        chunk_size_s = 30.0
        chunk_durations = [30.0, 30.0, 30.0]

        for idx in range(len(chunk_durations)):
            old_start = self._old_buggy_start_time(idx, chunk_size_s)
            new_start = self._new_accumulated_start_time(chunk_durations, idx)
            assert math.isclose(old_start, new_start, abs_tol=1e-9), (
                f"Chunk {idx}: With equal chunks, old and new should agree"
            )

    def test_drift_grows_linearly_with_old_logic(self):
        """Each chunk that is shorter than nominal adds to the drift."""
        chunk_size_s = 30.0
        chunk_durations = [29.5] * 10

        actual_start = 0.0
        for idx, dur in enumerate(chunk_durations):
            if idx > 0:
                old_start = self._old_buggy_start_time(idx, chunk_size_s)
                expected_drift = idx * 0.5
                actual_drift = old_start - actual_start
                assert math.isclose(actual_drift, expected_drift, abs_tol=1e-9), (
                    f"Chunk {idx}: drift should be {expected_drift}s, "
                    f"got {actual_drift}s"
                )
            actual_start += dur


# ---------------------------------------------------------------------------
# _split_audio tests (using standalone copies of the split logic)
# ---------------------------------------------------------------------------
class TestSplitAudio:
    """Test the audio chunking logic in isolation."""

    def test_short_audio_not_split(self):
        """Audio shorter than max_audio_clip_s produces one chunk."""
        cfg = _MockConfig(max_audio_clip_s=30)
        sr = int(cfg.sample_rate)

        audio = np.random.randn(10 * sr).astype(np.float32)
        chunks = _split_audio(audio, sr, cfg)

        assert len(chunks) == 1
        assert len(chunks[0]) == len(audio)

    def test_long_audio_produces_multiple_chunks(self):
        """Audio longer than max_audio_clip_s is split."""
        cfg = _MockConfig(max_audio_clip_s=30)
        sr = int(cfg.sample_rate)

        audio = np.random.randn(95 * sr).astype(np.float32)
        chunks = _split_audio(audio, sr, cfg)

        assert len(chunks) >= 3
        total_samples = sum(len(c) for c in chunks)
        assert total_samples == len(audio)

    def test_chunks_have_variable_length(self):
        """Chunks have variable length due to silence-based splitting."""
        cfg = _MockConfig(max_audio_clip_s=30, overlap_chunk_second=1)
        sr = int(cfg.sample_rate)

        duration_s = 120
        audio = np.random.randn(duration_s * sr).astype(np.float32)

        # Insert silence at specific points within the overlap windows
        silence_positions_s = [29.2, 58.7, 88.1]
        for pos in silence_positions_s:
            start = int(pos * sr)
            end = min(start + cfg.min_energy_split_window_size, len(audio))
            audio[start:end] = 0.0

        chunks = _split_audio(audio, sr, cfg)
        chunk_durations = [len(c) / sr for c in chunks]

        unique_durations = set(round(d, 2) for d in chunk_durations)
        assert len(unique_durations) > 1, (
            f"Expected variable chunk durations, got: {chunk_durations}"
        )

    def test_total_samples_preserved(self):
        """No samples lost or duplicated during splitting."""
        cfg = _MockConfig(max_audio_clip_s=30)
        sr = int(cfg.sample_rate)

        audio = np.random.randn(90 * sr).astype(np.float32)
        chunks = _split_audio(audio, sr, cfg)

        total = sum(len(c) for c in chunks)
        assert total == len(audio), f"Total samples {total} != original {len(audio)}"


# ---------------------------------------------------------------------------
# End-to-end regression: _split_audio + accumulated timestamps
# ---------------------------------------------------------------------------
class TestEndToEndTimestamps:
    """Combines _split_audio with timestamp calculation to verify
    the full pipeline produces accurate timestamps."""

    def test_accumulated_timestamps_match_actual_chunk_positions(self):
        """After splitting real audio, accumulated chunk durations
        produce timestamps matching the actual position in the audio."""
        cfg = _MockConfig(max_audio_clip_s=30, overlap_chunk_second=1)
        sr = int(cfg.sample_rate)

        # 2-minute audio with silence at non-uniform positions
        audio = np.random.randn(120 * sr).astype(np.float32)
        for pos_s in [29.3, 59.1, 88.8]:
            start = int(pos_s * sr)
            end = min(start + cfg.min_energy_split_window_size, len(audio))
            audio[start:end] = 0.0

        chunks = _split_audio(audio, sr, cfg)
        chunk_durations = [float(len(c)) / float(sr) for c in chunks]

        # Verify accumulated timestamps
        accumulated = 0.0
        expected_sample_offset = 0
        for idx, chunk in enumerate(chunks):
            # The accumulated time should match the actual sample offset
            actual_time = expected_sample_offset / sr
            assert math.isclose(accumulated, actual_time, abs_tol=1e-6), (
                f"Chunk {idx}: accumulated={accumulated:.6f}s, "
                f"actual={actual_time:.6f}s"
            )

            accumulated += chunk_durations[idx]
            expected_sample_offset += len(chunk)

    def test_old_logic_drifts_on_real_split(self):
        """The old idx * chunk_size logic drifts on real split audio."""
        cfg = _MockConfig(max_audio_clip_s=30, overlap_chunk_second=1)
        sr = int(cfg.sample_rate)
        chunk_size_s = float(cfg.max_audio_clip_s)

        # Create audio with silence causing variable chunks
        audio = np.random.randn(120 * sr).astype(np.float32)
        for pos_s in [29.3, 59.1, 88.8]:
            start = int(pos_s * sr)
            end = min(start + cfg.min_energy_split_window_size, len(audio))
            audio[start:end] = 0.0

        chunks = _split_audio(audio, sr, cfg)

        # Check if there's actual variability (precondition)
        durations = [len(c) / sr for c in chunks]
        has_variable_chunks = len(set(round(d, 2) for d in durations)) > 1
        if not has_variable_chunks:
            pytest.skip("No variable chunks produced (silence might not work)")

        # Verify old logic drifts
        actual_start = 0.0
        found_drift = False
        for idx, chunk in enumerate(chunks):
            old_start = float(idx * chunk_size_s)
            if idx > 0 and abs(old_start - actual_start) > 0.01:
                found_drift = True
                break
            actual_start += len(chunk) / sr

        assert found_drift, "Expected drift with old logic on variable-length chunks"
