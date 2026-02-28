# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from io import BytesIO
from pathlib import Path

import av
import librosa
import numpy as np
import numpy.typing as npt
import pybase64
import soundfile
import torch

from vllm.utils.serial_utils import tensor2base64

from .base import MediaIO


def extract_audio_from_video_bytes(
    data: bytes,
    sr: float | None = None,
) -> tuple[npt.NDArray, float]:
    """Extract the audio track from raw video bytes using PyAV.

    PyAV wraps FFmpeg's C libraries in-process — no subprocess is
    spawned, which is critical to avoid crashing CUDA-active vLLM
    worker processes.

    Args:
        data: Raw video file bytes (e.g. from an mp4 file).
        sr: Target sampling rate.  If ``None``, the native rate is used.

    Returns:
        A tuple of ``(waveform, sample_rate)`` suitable for use as an
        :class:`AudioItem`.
    """
    if data is None or len(data) == 0:
        raise ValueError(
            "Cannot extract audio: video bytes are missing or empty. "
            "Ensure video was loaded with keep_video_bytes=True for "
            "audio-in-video extraction."
        )
    try:
        with av.open(BytesIO(data)) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream found in the video.")
            stream = container.streams.audio[0]
            native_sr = stream.rate

            chunks: list[npt.NDArray] = []
            for frame in container.decode(audio=0):
                # to_ndarray() returns shape (channels, samples) for planar
                # formats and (1, samples) for packed formats.
                arr = frame.to_ndarray()
                chunks.append(arr.mean(axis=0) if arr.ndim > 1 else arr)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            "Invalid or corrupted video data when extracting audio. "
            "Ensure the input is valid video bytes (e.g. a complete MP4)."
        ) from e

    if not chunks:
        raise ValueError("No audio found in the video.")
    else:
        audio = np.concatenate(chunks).astype(np.float32)

    if sr is not None and sr != native_sr:
        audio = librosa.resample(audio, orig_sr=float(native_sr), target_sr=float(sr))
        return audio, float(sr)

    return audio, float(native_sr)


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
