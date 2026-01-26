# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from io import BytesIO
from pathlib import Path
import wave

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

_PCM16_SCALE = float(2**15)
_PCM24_SCALE = float(2**23)
_PCM32_SCALE = float(2**31)


def _load_wav_bytes(data: bytes) -> tuple[npt.NDArray[np.floating], float]:
    with wave.open(BytesIO(data), "rb") as wf:
        comptype = wf.getcomptype()
        if comptype != "NONE":
            raise ValueError(
                f"Unsupported WAV compression type: {comptype}. "
                "Only uncompressed PCM (comptype='NONE') is supported. "
                "For compressed formats (ULAW, ALAW, etc.), use FLAC/OGG instead."
            )
        sr = float(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        # 8-bit PCM is unsigned.
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _PCM16_SCALE
    elif sampwidth == 3:
        # 24-bit PCM is common; decode little-endian signed integers.
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        audio_i32 = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        # Sign-extend 24-bit to 32-bit.
        audio_i32 = np.where(audio_i32 & 0x800000, audio_i32 - 0x1000000, audio_i32)
        audio = audio_i32.astype(np.float32) / _PCM24_SCALE
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / _PCM32_SCALE
    else:
        raise ValueError(f"Unsupported wav sampwidth={sampwidth} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sr


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
        if not isinstance(librosa, PlaceholderModule):
            return librosa.load(BytesIO(data), sr=None)

        # Minimal fallback to avoid hard dependency on librosa for WAV inputs.
        # For non-WAV formats, users must install librosa (or provide audio
        # embeddings directly).
        try:
            return _load_wav_bytes(data)
        except Exception as e:
            raise RuntimeError(
                "Failed to load audio bytes without librosa. "
                "Install librosa to support non-WAV audio formats."
            ) from e

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        if not isinstance(librosa, PlaceholderModule):
            return librosa.load(filepath, sr=None)

        try:
            data = filepath.read_bytes()
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file: {filepath}") from e

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: tuple[npt.NDArray, int],
        *,
        audio_format: str = "WAV",
    ) -> str:
        audio, sr = media

        if isinstance(soundfile, PlaceholderModule):
            raise RuntimeError(
                "soundfile is required to encode audio to base64. "
                "Install soundfile or provide pre-encoded bytes."
            )

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
