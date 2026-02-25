# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]


try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = PlaceholderModule("scipy").placeholder_attr("signal")  # type: ignore[assignment]

# ============================================================


class ChannelReduction(str, Enum):
    """Method to reduce multi-channel audio to target channels."""

    MEAN = "mean"  # Average across channels (default, preserves energy balance)
    FIRST = "first"  # Take first channel only
    MAX = "max"  # Take max value across channels
    SUM = "sum"  # Sum across channels


@dataclass
class AudioSpec:
    """Specification for target audio format.

    This dataclass defines the expected audio format for a model's feature
    extractor. It is used to normalize audio data before processing.

    Attributes:
        target_channels: Number of output channels. None means passthrough
            (no normalization). 1 = mono, 2 = stereo, etc.
        channel_reduction: Method to reduce channels when input has more
            channels than target. Only used when reducing channels.
    """

    target_channels: int | None = 1
    channel_reduction: ChannelReduction = ChannelReduction.MEAN

    @property
    def needs_normalization(self) -> bool:
        """Whether audio normalization is needed."""
        return self.target_channels is not None

    def __repr__(self) -> str:
        if self.target_channels is None:
            return "AudioSpec(passthrough)"
        return (
            f"AudioSpec(channels={self.target_channels}, "
            f"reduction={self.channel_reduction.value})"
        )


# Pre-defined specs for common use cases
MONO_AUDIO_SPEC = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
PASSTHROUGH_AUDIO_SPEC = AudioSpec(target_channels=None)


def normalize_audio(
    audio: npt.NDArray[np.floating] | torch.Tensor,
    spec: AudioSpec,
) -> npt.NDArray[np.floating] | torch.Tensor:
    """Normalize audio to the specified format.

    This function handles channel reduction for multi-channel audio,
    supporting both numpy arrays and torch tensors.

    Args:
        audio: Input audio data. Can be:
            - 1D array/tensor: (time,) - already mono
            - 2D array/tensor: (channels, time) - standard format from torchaudio
            - 2D array/tensor: (time, channels) - format from soundfile
              (will be auto-detected and transposed if time > channels)
        spec: AudioSpec defining the target format.

    Returns:
        Normalized audio in the same type as input (numpy or torch).
        For mono output (target_channels=1), returns 1D array/tensor.

    Raises:
        ValueError: If audio has unsupported dimensions or channel expansion
            is requested (e.g., mono to stereo).
    """
    if not spec.needs_normalization:
        return audio

    # Handle 1D audio (already mono)
    if audio.ndim == 1:
        if spec.target_channels == 1:
            return audio
        raise ValueError(f"Cannot expand mono audio to {spec.target_channels} channels")

    # Handle 2D audio
    if audio.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {audio.shape}. Expected 1D or 2D.")

    # Auto-detect format: if shape[0] > shape[1], assume (time, channels)
    # This handles soundfile format where time dimension is typically much larger
    if audio.shape[0] > audio.shape[1]:
        # Transpose from (time, channels) to (channels, time)
        audio = audio.T if isinstance(audio, np.ndarray) else audio.T

    num_channels = audio.shape[0]

    # No reduction needed if already at target
    if num_channels == spec.target_channels:
        return audio

    # Cannot expand channels
    if num_channels < spec.target_channels:
        raise ValueError(
            f"Cannot expand {num_channels} channels to {spec.target_channels}"
        )

    # Reduce channels
    is_numpy = isinstance(audio, np.ndarray)

    if spec.target_channels == 1:
        # Reduce to mono
        if spec.channel_reduction == ChannelReduction.MEAN:
            result = np.mean(audio, axis=0) if is_numpy else audio.mean(dim=0)
        elif spec.channel_reduction == ChannelReduction.FIRST:
            result = audio[0]
        elif spec.channel_reduction == ChannelReduction.MAX:
            result = np.max(audio, axis=0) if is_numpy else audio.max(dim=0).values
        elif spec.channel_reduction == ChannelReduction.SUM:
            result = np.sum(audio, axis=0) if is_numpy else audio.sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction method: {spec.channel_reduction}")
        return result
    else:
        # Reduce to N channels (take first N and apply reduction if needed)
        # For now, just take first N channels
        return audio[: spec.target_channels]


# ============================================================
# Audio Resampling
# ============================================================


def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    if orig_sr > target_sr:
        return scipy_signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy_signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio


class AudioResampler:
    """Resample audio data to a target sample rate."""

    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["librosa", "scipy"] = "librosa",
    ):
        self.target_sr = target_sr
        self.method = method

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        if self.target_sr is None:
            raise RuntimeError(
                "Audio resampling is not supported when `target_sr` is not provided"
            )
        if math.isclose(
            float(orig_sr),
            float(self.target_sr),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            return audio
        if self.method == "librosa":
            return resample_audio_librosa(
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        elif self.method == "scipy":
            return resample_audio_scipy(
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        else:
            raise ValueError(
                f"Invalid resampling method: {self.method}. "
                "Supported methods are 'librosa' and 'scipy'."
            )


# ============================================================
# Audio Chunking / Splitting
# ============================================================


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
