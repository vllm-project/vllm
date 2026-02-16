# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio processing utilities for vLLM."""

import math

import numpy as np


def split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    max_clip_duration_s: float,
    overlap_duration_s: float,
    min_energy_window_size: int,
) -> list[np.ndarray]:
    """Split audio into chunks with intelligent split points.

    Splits long audio into smaller chunks at low-energy regions to minimize
    cutting through speech. Uses overlapping windows to find quiet moments
    for splitting.

    Args:
        audio_data: Audio array to split. Can be 1D (mono) or multi-dimensional.
                   Splits along the last dimension (time axis).
        sample_rate: Sample rate of the audio in Hz.
        max_clip_duration_s: Maximum duration of each chunk in seconds.
        overlap_duration_s: Overlap duration in seconds between consecutive chunks.
                           Used to search for optimal split points.
        min_energy_window_size: Window size in samples for finding low-energy regions.

    Returns:
        List of audio chunks. Each chunk is a numpy array with the same shape
        as the input except for the last (time) dimension.

    Example:
        >>> audio = np.random.randn(1040000)  # 65 seconds at 16kHz
        >>> chunks = split_audio(
        ...     audio_data=audio,
        ...     sample_rate=16000,
        ...     max_clip_duration_s=30.0,
        ...     overlap_duration_s=1.0,
        ...     min_energy_window_size=1600,
        ... )
        >>> len(chunks)
        3
    """
    chunk_size = int(sample_rate * max_clip_duration_s)
    overlap_size = int(sample_rate * overlap_duration_s)
    chunks = []
    i = 0

    while i < audio_data.shape[-1]:
        if i + chunk_size >= audio_data.shape[-1]:
            # Handle last chunk - take everything remaining
            chunks.append(audio_data[..., i:])
            break

        # Find the best split point in the overlap region
        search_start = i + chunk_size - overlap_size
        search_end = min(i + chunk_size, audio_data.shape[-1])
        split_point = find_split_point(
            audio_data, search_start, search_end, min_energy_window_size
        )

        # Extract chunk up to the split point
        chunks.append(audio_data[..., i:split_point])
        i = split_point

    return chunks


def find_split_point(
    wav: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window: int,
) -> int:
    """Find the best point to split audio by looking for silence or low amplitude.

    Searches for the quietest region within a specified range by calculating
    RMS energy in sliding windows.

    Args:
        wav: Audio array. Can be 1D or multi-dimensional.
        start_idx: Start index of search region (inclusive).
        end_idx: End index of search region (exclusive).
        min_energy_window: Window size in samples for energy calculation.

    Returns:
        Index of the quietest point within the search region. This is the
        recommended split point to minimize audio artifacts.

    Example:
        >>> audio = np.random.randn(32000)
        >>> # Insert quiet region
        >>> audio[16000:17600] = 0.01
        >>> split_idx = find_split_point(
        ...     wav=audio,
        ...     start_idx=0,
        ...     end_idx=32000,
        ...     min_energy_window=1600,
        ... )
        >>> 16000 <= split_idx <= 17600
        True
    """
    segment = wav[start_idx:end_idx]

    # Calculate RMS energy in small windows
    min_energy = math.inf
    quietest_idx = 0

    for i in range(0, len(segment) - min_energy_window, min_energy_window):
        window = segment[i : i + min_energy_window]
        energy = (window**2).mean() ** 0.5
        if energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy

    return quietest_idx
