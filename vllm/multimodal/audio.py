# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pybase64
import torch

from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.serial_utils import tensor2base64

from .base import MediaIO

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


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


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return librosa.load(BytesIO(data), sr=None)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return librosa.load(filepath, sr=None)

    def encode_base64(
        self,
        media: tuple[npt.NDArray, int],
        *,
        audio_format: str = "WAV",
    ) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format=audio_format)
            data = buffer.getvalue()

        return base64.b64encode(data).decode("utf-8")


class AudioEmbeddingMediaIO(MediaIO[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
