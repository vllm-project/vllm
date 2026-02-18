# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path

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


def extract_audio_from_video_bytes(
    data: bytes,
    sr: float | None = None,
    max_duration: float | None = None,
) -> tuple[npt.NDArray, float]:
    """Extract the audio track from raw video bytes using librosa.

    Args:
        data: Raw video file bytes (e.g. from an mp4 file).
        sr: Target sampling rate. If ``None``, the native rate is used.
        max_duration: If set, only load the first *max_duration* seconds
            of audio.

    Returns:
        A tuple of ``(waveform, sample_rate)`` suitable for use as an
        :class:`AudioItem`.
    """
    # MP4 containers require file-path access (soundfile can't read them
    # from BytesIO).  Write to a temp file and let librosa's audioread
    # backend handle the container -- no subprocess is spawned, which is
    # critical to avoid crashing CUDA-active vLLM worker processes.
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=tmp_dir
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return librosa.load(tmp_path, sr=sr, duration=max_duration)
    finally:
        os.unlink(tmp_path)


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
