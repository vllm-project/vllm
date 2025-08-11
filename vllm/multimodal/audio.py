# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import MediaIO

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


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
    # lazy import scipy.signal, otherwise it will crash doc build.
    import scipy.signal

    if orig_sr > target_sr:
        return scipy.signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy.signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio


class AudioResampler:
    """Resample audio data to a target sample rate."""

    def __init__(
        self,
        target_sr: Optional[float] = None,
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
            raise RuntimeError("Audio resampling is not supported when "
                               "`target_sr` is not provided")
        if self.method == "librosa":
            return resample_audio_librosa(audio,
                                          orig_sr=orig_sr,
                                          target_sr=self.target_sr)
        elif self.method == "scipy":
            return resample_audio_scipy(audio,
                                        orig_sr=orig_sr,
                                        target_sr=self.target_sr)
        else:
            raise ValueError(f"Invalid resampling method: {self.method}. "
                             "Supported methods are 'librosa' and 'scipy'.")


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

    def encode_base64(self, media: tuple[npt.NDArray, float]) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format="WAV")
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')
