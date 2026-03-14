# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64
import torch

from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.serial_utils import tensor2base64

from .base import MediaIO

try:
    import av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


def load_audio_pyav(source: BytesIO | Path) -> tuple[npt.NDArray, float]:
    """Load an audio file using PyAV (FFmpeg), returning float32 mono waveform.

    Decodes the audio stream at its native sample rate. Channel reduction to
    mono is performed by averaging across channels.  Resampling to a
    model-specific rate is left to the downstream :class:`AudioResampler`.

    Args:
        source: A :class:`~io.BytesIO` buffer or a filesystem
            :class:`~pathlib.Path`.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a 1-D float32
        NumPy array and *sample_rate* is the native sample rate in Hz.
    """
    with av.open(source) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream found.")
        stream = container.streams.audio[0]
        native_sr = stream.rate

        # Normalize to float32 planar at the native sample rate so that
        # `to_ndarray()` always returns a consistent (channels, samples) array.
        resampler = av.AudioResampler(format="fltp", rate=native_sr)

        chunks: list[npt.NDArray] = []
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                chunks.append(out_frame.to_ndarray())
        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray())

    if not chunks:
        raise ValueError("No audio data found in the file.")

    # (channels, samples) → average to mono → (samples,)
    audio = np.concatenate(chunks, axis=-1)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio.astype(np.float32), float(native_sr)


def extract_audio_from_video_bytes(
    data: bytes,
) -> tuple[npt.NDArray, float]:
    """Extract the audio track from raw video bytes using PyAV.

    PyAV wraps FFmpeg's C libraries in-process — no subprocess is
    spawned, which is critical to avoid crashing CUDA-active vLLM
    worker processes.

    The returned waveform is at the native sample rate of the video's
    audio stream.  Resampling to a model-specific rate is left to the
    downstream :class:`AudioResampler` in the parsing pipeline.

    Args:
        data: Raw video file bytes (e.g. from an mp4 file).

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

    audio = np.concatenate(chunks).astype(np.float32)
    return audio, float(native_sr)


def is_video(data: bytes) -> bool:
    """Check if the fetched bytes are video"""
    if len(data) < 12:
        return False

    box_type = data[4:8]
    major_brand = data[8:12]

    MP4_BRANDS = {
        b"mp41",
        b"mp42",  # MP4
        b"isom",  # ISO Base Media
        b"iso2",
        b"iso4",
        b"iso5",
        b"iso6",
        b"M4V ",
        b"M4A ",  # Apple
        b"avc1",  # H.264
        b"dash",  # DASH
        b"mmp4",
        b"MSNV",
    }

    is_avi = data[:4] == b"RIFF" and major_brand == b"AVI "
    is_mp4 = box_type == b"ftyp" and major_brand in MP4_BRANDS
    return is_mp4 or is_avi


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        if is_video(data):
            return extract_audio_from_video_bytes(data)
        return load_audio_pyav(BytesIO(data))

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return load_audio_pyav(filepath)

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
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

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
