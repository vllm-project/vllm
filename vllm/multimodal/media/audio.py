# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.multimodal.audio import resample_audio_pyav
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import check_sparse_tensor_invariants_threadsafe

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

try:
    from torchcodec.decoders import AudioDecoder
except (ImportError, RuntimeError):
    # RuntimeError: torchcodec is installed but the system ffmpeg is missing.
    AudioDecoder = PlaceholderModule("torchcodec").placeholder_attr(  # type: ignore[assignment]
        "decoders.AudioDecoder"
    )


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

# Audio decoding backends selectable via `load_audio(backend=...)`,
# `--media-io-kwargs '{"audio": {"audio_backend": ...}}'`, or
# VLLM_AUDIO_LOADER_BACKEND. "auto" tries torchcodec, then soundfile,
# then PyAV.
AUDIO_BACKENDS = ("auto", "soundfile", "pyav", "torchcodec")

# Slack on the torchcodec decode window when enforcing `max_duration_s`, so an
# over-long stream is rejected rather than truncated to the limit.
_DURATION_GUARD_MARGIN_S = 1.0


def load_audio_pyav(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[npt.NDArray, float]:
    """Load an audio file using PyAV (FFmpeg), returning float32 mono waveform.

    Decodes the audio stream at its native sample rate. Channel reduction to
    mono is performed by averaging across channels.  Resampling to a
    model-specific rate is left to the downstream :class:`AudioResampler`.

    Args:
        path: A :class:`~io.BytesIO` buffer, a filesystem
            :class:`~pathlib.Path`, or a string path.
        max_duration_s: If set, abort decoding once the accumulated
            sample count exceeds this many seconds of audio.  Prevents
            decompression-bomb attacks where a small compressed file
            expands into gigabytes of PCM.

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

            # Early rejection from container/stream metadata to avoid
            # wasting resources on decoding decompression bombs.
            if max_duration_s is not None:
                metadata_duration_s = None
                if stream.duration and stream.time_base:
                    metadata_duration_s = float(stream.duration * stream.time_base)
                elif container.duration:
                    metadata_duration_s = container.duration / 1_000_000
                if (
                    metadata_duration_s is not None
                    and metadata_duration_s > max_duration_s
                ):
                    raise ValueError(
                        f"Audio exceeds maximum allowed duration of "
                        f"{max_duration_s}s (metadata reports "
                        f"{metadata_duration_s:.1f}s). Set "
                        f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                        f"increase this limit."
                    )

            max_samples = (
                int(sr * max_duration_s) if max_duration_s is not None else None
            )
            total_samples = 0
            total_decode_bytes = 0

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
                        arr = out_frame.to_ndarray()
                        total_samples += arr.shape[-1]
                        total_decode_bytes += arr.nbytes
                        chunks.append(arr)
                else:
                    arr = frame.to_ndarray()
                    total_samples += arr.shape[-1]
                    total_decode_bytes += arr.nbytes
                    chunks.append(arr)

                if max_samples is not None and total_samples > max_samples:
                    raise ValueError(
                        f"Audio exceeds maximum allowed duration of "
                        f"{max_duration_s}s (decoded {total_samples} "
                        f"samples at {sr}Hz). Set "
                        f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                        f"increase this limit."
                    )
                if (
                    max_decode_bytes is not None
                    and total_decode_bytes > max_decode_bytes
                ):
                    raise ValueError(
                        f"Audio decode exceeded "
                        f"{max_decode_bytes / MiB_bytes:.0f} MiB memory "
                        f"limit ({total_decode_bytes / MiB_bytes:.0f} MiB "
                        f"decoded so far). Set "
                        f"VLLM_MAX_AUDIO_DECODE_BYTES to increase this "
                        f"limit."
                    )
    except (ValueError, ImportError):
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
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio via soundfile"""
    with soundfile.SoundFile(path) as f:
        native_sr = f.samplerate
        if max_duration_s is not None:
            file_duration_s = f.frames / native_sr
            if file_duration_s > max_duration_s:
                raise ValueError(
                    f"Audio exceeds maximum allowed duration of "
                    f"{max_duration_s}s (file contains "
                    f"{file_duration_s:.1f}s at {native_sr}Hz). Set "
                    f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                    f"increase this limit."
                )
        if max_decode_bytes is not None:
            estimated_bytes = f.frames * f.channels * np.dtype(np.float32).itemsize
            if estimated_bytes > max_decode_bytes:
                raise ValueError(
                    f"Audio would allocate {estimated_bytes / MiB_bytes:.0f} "
                    f"MiB of PCM ({f.frames} frames x {f.channels} channels"
                    f" x 4B), exceeding the "
                    f"{max_decode_bytes / MiB_bytes:.0f} MiB limit. Set "
                    f"VLLM_MAX_AUDIO_DECODE_BYTES to increase this limit."
                )
        y = f.read(dtype="float32", always_2d=False).T

    if mono and y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))

    if sr is not None and sr != native_sr:
        y = resample_audio_pyav(y, orig_sr=native_sr, target_sr=sr)
        return y, int(sr)
    return y, native_sr


def load_audio_torchcodec(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[npt.NDArray, float]:
    """Load an audio file using torchcodec, returning a float32 waveform.

    Unlike :func:`load_audio_pyav`, which drives FFmpeg through a per-frame
    Python generator, torchcodec decodes inside a single call that releases
    the GIL. Resampling also happens in C++ (``sample_rate=sr``), so this
    backend never falls back to PyAV or libsndfile.

    Args:
        path: A :class:`~io.BytesIO` buffer, a filesystem
            :class:`~pathlib.Path`, or a string path.
        sr: Output sample rate; ``None`` keeps the native rate. Must be an
            integer when set.
        mono: Average channels down to a single channel.
        max_duration_s: Reject audio longer than this many seconds.
            Prevents decompression-bomb attacks where a small compressed
            file expands into gigabytes of PCM.
        max_decode_bytes: Reject audio whose decoded PCM exceeds this many
            bytes.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a float32 NumPy
        array (1-D when ``mono=True``) and *sample_rate* is the output
        sample rate in Hz.
    """
    try:
        # AudioDecoder takes an integer sample_rate; reject a non-integer
        # `sr` here rather than surfacing a TypeError from the C++ layer.
        if sr is not None and sr != int(sr):
            raise ValueError(f"torchcodec requires an integer sample rate, got {sr}")
        decoder = AudioDecoder(path, sample_rate=int(sr) if sr is not None else None)

        # Pre-check the memory budget from container metadata so an
        # obviously oversized stream is rejected before allocating the PCM
        # buffer. Metadata is attacker-controlled, so this may only reject,
        # never admit; the post-decode check below is authoritative.
        if max_decode_bytes is not None:
            metadata = decoder.metadata
            est_duration_s = metadata.duration_seconds
            # Estimate at the *output* rate: when `sr` is set torchcodec
            # resamples to it during decode.
            est_sample_rate = sr if sr is not None else metadata.sample_rate
            est_num_channels = metadata.num_channels
            if (
                est_duration_s is not None
                and est_sample_rate is not None
                and est_num_channels is not None
            ):
                estimated_bytes = (
                    int(est_duration_s * est_sample_rate)
                    * est_num_channels
                    * np.dtype(np.float32).itemsize
                )
                if estimated_bytes > max_decode_bytes:
                    raise ValueError(
                        f"Audio would allocate "
                        f"{estimated_bytes / MiB_bytes:.0f} MiB of PCM "
                        f"(~{est_duration_s:.1f}s at {est_sample_rate}Hz x "
                        f"{est_num_channels}ch), exceeding the "
                        f"{max_decode_bytes / MiB_bytes:.0f} MiB limit. Set "
                        f"VLLM_MAX_AUDIO_DECODE_BYTES to increase this limit."
                    )

        if max_duration_s is None:
            samples = decoder.get_all_samples()
        else:
            # Guard in two stages like load_audio_pyav: the metadata
            # pre-check may only reject, never admit.
            duration_s = decoder.metadata.duration_seconds
            if duration_s is not None and duration_s > max_duration_s:
                raise ValueError(
                    f"Audio exceeds maximum allowed duration of "
                    f"{max_duration_s}s (metadata reports "
                    f"{duration_s:.1f}s). Set "
                    f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                    f"increase this limit."
                )
            # Bounding the range keeps an under-reported duration from
            # expanding into unbounded PCM; decoding slightly past the
            # limit makes an over-long stream fail the check below instead
            # of being silently truncated to it.
            samples = decoder.get_samples_played_in_range(
                0.0, max_duration_s + _DURATION_GUARD_MARGIN_S
            )
            # Same `int()` semantics as load_audio_pyav's
            # `total_samples > int(sr * max_duration_s)`.
            max_samples = int(samples.sample_rate * max_duration_s)
            if samples.data.shape[-1] > max_samples:
                raise ValueError(
                    f"Audio exceeds maximum allowed duration of "
                    f"{max_duration_s}s (decoded {samples.data.shape[-1]} "
                    f"samples at {samples.sample_rate}Hz). Set "
                    f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                    f"increase this limit."
                )

        # Re-check the actual decoded size: the metadata estimate above can
        # be bypassed by a container that lies about its duration.
        if max_decode_bytes is not None and samples.data.nbytes > max_decode_bytes:
            raise ValueError(
                f"Audio decode exceeded "
                f"{max_decode_bytes / MiB_bytes:.0f} MiB memory "
                f"limit ({samples.data.nbytes / MiB_bytes:.0f} MiB decoded). "
                f"Set VLLM_MAX_AUDIO_DECODE_BYTES to increase this limit."
            )
    except (ValueError, ImportError):
        raise
    except Exception as e:
        raise ValueError(
            "Invalid or corrupted audio data. Ensure the input is a valid "
            "audio or video file (e.g. a complete WAV, MP3, or MP4)."
        ) from e

    out_sr = samples.sample_rate
    audio = samples.data.numpy()  # (num_channels, num_samples), float32
    if audio.size == 0:
        raise ValueError("No audio found in the input.")

    if mono and audio.ndim > 1:
        # Same reduction as load_audio_pyav, so both backends agree at the
        # native sample rate, which is what AudioMediaIO requests.
        audio = np.mean(audio, axis=0)

    return audio, out_sr


def load_audio(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
    backend: str | None = None,
):
    """Load audio using the selected decoding backend.

    Args:
        backend: One of ``AUDIO_BACKENDS``. ``None`` (default) resolves to
            ``VLLM_AUDIO_LOADER_BACKEND``. ``"auto"`` tries torchcodec,
            then falls back to the soundfile → PyAV chain; the other values
            select a single backend with no fallback.
    """
    backend = backend or envs.VLLM_AUDIO_LOADER_BACKEND
    if backend not in AUDIO_BACKENDS:
        raise ValueError(
            f"Unknown audio backend {backend!r}. "
            f"Available backends: {list(AUDIO_BACKENDS)}"
        )
    if backend == "torchcodec":
        return load_audio_torchcodec(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
    if backend == "pyav":
        return load_audio_pyav(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
    if backend == "soundfile":
        return load_audio_soundfile(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )

    # "auto": torchcodec first, then the soundfile → PyAV chain.
    try:
        return load_audio_torchcodec(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
    except ImportError as exc:
        # torchcodec (or its system ffmpeg) is not installed — fall through
        # to soundfile. Decode errors do not fall back: torchcodec is
        # FFmpeg-based like PyAV, and guard ValueErrors must propagate.
        logger.warning(
            "torchcodec unavailable (%r); falling back to soundfile/PyAV.", exc
        )
        if isinstance(path, BytesIO):
            path.seek(0)
    try:
        return load_audio_soundfile(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
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
        return load_audio_pyav(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
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

        # Allow per-server override of the audio decoding backend, e.g.:
        #   --media-io-kwargs '{"audio": {"audio_backend": "torchcodec"}}'
        # Overrides the global VLLM_AUDIO_LOADER_BACKEND env var. Validate
        # eagerly so a misconfigured deployment fails at startup rather
        # than per request.
        backend = kwargs.pop("audio_backend", None)
        if backend is not None and backend not in AUDIO_BACKENDS:
            raise ValueError(
                f"Unknown audio_backend {backend!r}. "
                f"Available backends: {list(AUDIO_BACKENDS)}"
            )
        self.audio_backend = backend

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return load_audio(
            BytesIO(data),
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
            max_decode_bytes=envs.VLLM_MAX_AUDIO_DECODE_BYTES,
            backend=self.audio_backend,
        )

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return load_audio(
            filepath,
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
            max_decode_bytes=envs.VLLM_MAX_AUDIO_DECODE_BYTES,
            backend=self.audio_backend,
        )

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
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
