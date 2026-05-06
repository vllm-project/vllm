# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64
import torch

from vllm.logger import init_logger
from vllm.multimodal.audio import resample_audio_pyav
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.serial_utils import tensor2base64

from .base import MediaIO

logger = init_logger(__name__)

try:
    import av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


# Public libsndfile error codes exposed via `soundfile.LibsndfileError.code`,
# soundfile being the main audio loading backend. Used to validate if an audio
# loading error is due to a server error vs a client error (invalid audio file).
# 0 = sf_error(NULL) race condition: when multiple threads fail sf_open_virtual
#     concurrently, one thread may clear the global error before another reads it,
#     producing code=0 ("Garbled error message from libsndfile" in soundfile).
#     See: https://github.com/bastibe/python-soundfile/issues/479
# 1 = unrecognised format      (file is not a supported audio container)
# 3 = malformed file           (corrupt or structurally invalid audio)
# 4 = unsupported encoding     (codec not supported by this libsndfile build)
_BAD_SF_CODES = {0, 1, 3, 4}


def load_audio_pyav(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
) -> tuple[npt.NDArray, float]:
    """Load an audio file using PyAV (FFmpeg), returning float32 mono waveform.

    Decodes the audio stream at its native sample rate. Channel reduction to
    mono is performed by averaging across channels.  Resampling to a
    model-specific rate is left to the downstream :class:`AudioResampler`.

    Args:
        path: A :class:`~io.BytesIO` buffer, a filesystem
            :class:`~pathlib.Path`, or a string path.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a 1-D float32
        NumPy array and *sample_rate* is the native sample rate in Hz.
    """
    native_sr = None
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream found.")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            native_sr = stream.rate
            sr = sr or native_sr

            chunks: list[npt.NDArray] = []
            needs_resampling = not math.isclose(
                float(sr),
                float(native_sr),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            resampler = (
                av.AudioResampler(format="fltp", layout="mono", rate=sr)
                if needs_resampling
                else None
            )
            for frame in container.decode(stream):
                if needs_resampling:
                    assert resampler is not None
                    for out_frame in resampler.resample(frame):
                        chunks.append(out_frame.to_ndarray())
                else:
                    chunks.append(frame.to_ndarray())
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            "Invalid or corrupted video data when extracting audio. "
            "Ensure the input is valid video bytes (e.g. a complete MP4)."
        ) from e

    if not chunks:
        raise ValueError("No audio found in the video.")

    audio = np.concatenate(chunks, axis=-1).astype(np.float32)
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    return audio, sr


def load_audio_soundfile(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio via soundfile"""
    with soundfile.SoundFile(path) as f:
        native_sr = f.samplerate
        y = f.read(dtype="float32", always_2d=False).T

    if mono and y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))

    if sr is not None and sr != native_sr:
        y = resample_audio_pyav(y, orig_sr=native_sr, target_sr=sr)
        return y, int(sr)
    return y, native_sr


def load_audio(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
):
    try:
        return load_audio_soundfile(path, sr=sr, mono=mono)
    except ImportError as exc:
        # soundfile (or resampy) is not installed — fall through to pyav.
        # NOTE: this clause must stay BEFORE ``soundfile.LibsndfileError``
        # because when soundfile is a PlaceholderModule, evaluating
        # ``soundfile.LibsndfileError`` itself raises ImportError.
        logger.error("Failed to load audio via soundfile: %r", exc)
    except soundfile.LibsndfileError as exc:
        # Only fall back for known format-detection failures.
        # Re-raise anything else (e.g. corrupt but recognised format).
        if exc.code not in _BAD_SF_CODES:
            raise
    # soundfile may have advanced the BytesIO seek position before failing;
    # reset it so PyAV can read from the beginning.
    if isinstance(path, BytesIO):
        path.seek(0)
    try:
        return load_audio_pyav(path, sr=sr, mono=mono)
    except ImportError:
        raise  # Let PlaceholderModule's message ("install vllm[audio]") propagate.
    except Exception as pyav_exc:
        raise ValueError("Invalid or unsupported audio file.") from pyav_exc


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
        return load_audio(BytesIO(data), sr=None)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return load_audio(filepath, sr=None)

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

        return pybase64.b64encode(data).decode("utf-8")


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
