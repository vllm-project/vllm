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
from vllm.utils.import_utils import PlaceholderModule, check_torchcodec_available
from vllm.utils.registry import ExtensionManager
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

# Slack on the torchcodec decode window when enforcing `max_duration_s`, so an
# over-long stream is rejected rather than truncated to the limit.
_DURATION_GUARD_MARGIN_S = 1.0


def load_audio_pyav(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
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
                        chunks.append(arr)
                else:
                    arr = frame.to_ndarray()
                    total_samples += arr.shape[-1]
                    chunks.append(arr)

                if max_samples is not None and total_samples > max_samples:
                    raise ValueError(
                        f"Audio exceeds maximum allowed duration of "
                        f"{max_duration_s}s (decoded {total_samples} "
                        f"samples at {sr}Hz). Set "
                        f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                        f"increase this limit."
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
) -> tuple[npt.NDArray, float]:
    """Load an audio file using torchcodec, returning a float32 waveform.

    Unlike :func:`load_audio_pyav`, which drives FFmpeg through a per-frame
    Python generator and so crosses the Python/C boundary once per 1024
    samples, torchcodec decodes inside a single call that releases the GIL.
    Concurrent requests therefore stop serialising on those crossings.

    Args:
        path: A :class:`~io.BytesIO` buffer, a filesystem
            :class:`~pathlib.Path`, or a string path.
        max_duration_s: If set, reject the input when the stream is longer
            than this many seconds. Prevents decompression-bomb attacks
            where a small compressed file expands into gigabytes of PCM.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a 1-D float32
        NumPy array (when ``mono``) and *sample_rate* is the sample rate
        in Hz.
    """
    try:
        # Pass `path` straight through: AudioDecoder reads file-like objects
        # on demand, so the encoded bytes are never copied.
        decoder = AudioDecoder(path)

        if max_duration_s is None:
            samples = decoder.get_all_samples()
        else:
            # Guard in two stages like load_audio_pyav. Container metadata is
            # attacker-controlled, so it may only reject, never admit.
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
            # expanding into unbounded PCM. Decoding slightly past the limit
            # is what makes an over-long stream fail the check below instead
            # of being silently truncated to it.
            samples = decoder.get_samples_played_in_range(
                0.0, max_duration_s + _DURATION_GUARD_MARGIN_S
            )
            decoded_duration_s = samples.data.shape[-1] / samples.sample_rate
            if decoded_duration_s > max_duration_s:
                raise ValueError(
                    f"Audio exceeds maximum allowed duration of "
                    f"{max_duration_s}s (decoded {decoded_duration_s:.1f}s "
                    f"at {samples.sample_rate}Hz). Set "
                    f"VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                    f"increase this limit."
                )
    except (ValueError, ImportError):
        raise
    except Exception as e:
        raise ValueError(
            "Invalid or corrupted video data when extracting audio. "
            "Ensure the input is valid video bytes (e.g. a complete MP4)."
        ) from e

    native_sr = samples.sample_rate
    audio = samples.data.numpy()  # (num_channels, num_samples), float32
    if audio.size == 0:
        raise ValueError("No audio found in the video.")

    if mono and audio.ndim > 1:
        # Same reduction as load_audio_pyav, so both backends agree at the
        # native sample rate, which is what AudioMediaIO requests.
        audio = np.mean(audio, axis=0)

    if sr is not None and not math.isclose(
        float(sr), float(native_sr), rel_tol=0.0, abs_tol=1e-6
    ):
        # Resampling after the full decode, whereas load_audio_pyav resamples
        # frame by frame during it, so the two do not agree sample-for-sample
        # on this path.
        audio = resample_audio_pyav(audio, orig_sr=native_sr, target_sr=sr)
        return audio, sr

    return audio, native_sr


def load_audio(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
):
    try:
        return load_audio_soundfile(
            path, sr=sr, mono=mono, max_duration_s=max_duration_s
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
        return load_audio_pyav(path, sr=sr, mono=mono, max_duration_s=max_duration_s)
    except ImportError:
        raise  # Let PlaceholderModule's message ("install vllm[audio]") propagate.
    except Exception as pyav_exc:
        raise ValueError("Invalid or unsupported audio file.") from pyav_exc


class AudioLoader:
    """Base class for audio decoding backends.

    Subclasses wrap one decoding library so that the backend can be chosen
    at runtime, mirroring :class:`~vllm.multimodal.video.VideoLoader`.
    """

    @classmethod
    def load_bytes(
        cls,
        path: BytesIO | Path | str,
        *,
        sr: float | None = None,
        mono: bool = True,
        max_duration_s: float | None = None,
    ) -> tuple[npt.NDArray, float]:
        raise NotImplementedError


AUDIO_LOADER_REGISTRY = ExtensionManager()


@AUDIO_LOADER_REGISTRY.register("auto")
class AutoAudioLoader(AudioLoader):
    """Default backend: soundfile, falling back to PyAV.

    This preserves the historical behaviour of :func:`load_audio` and stays
    the default so that existing deployments are unaffected.
    """

    @classmethod
    def load_bytes(
        cls,
        path: BytesIO | Path | str,
        *,
        sr: float | None = None,
        mono: bool = True,
        max_duration_s: float | None = None,
    ) -> tuple[npt.NDArray, float]:
        return load_audio(path, sr=sr, mono=mono, max_duration_s=max_duration_s)


@AUDIO_LOADER_REGISTRY.register("soundfile")
class SoundfileAudioLoader(AudioLoader):
    """Decode via soundfile (libsndfile), with no fallback."""

    @classmethod
    def load_bytes(
        cls,
        path: BytesIO | Path | str,
        *,
        sr: float | None = None,
        mono: bool = True,
        max_duration_s: float | None = None,
    ) -> tuple[npt.NDArray, float]:
        return load_audio_soundfile(
            path, sr=sr, mono=mono, max_duration_s=max_duration_s
        )


@AUDIO_LOADER_REGISTRY.register("pyav")
class PyAvAudioLoader(AudioLoader):
    """Decode via PyAV (FFmpeg), with no fallback."""

    @classmethod
    def load_bytes(
        cls,
        path: BytesIO | Path | str,
        *,
        sr: float | None = None,
        mono: bool = True,
        max_duration_s: float | None = None,
    ) -> tuple[npt.NDArray, float]:
        return load_audio_pyav(path, sr=sr, mono=mono, max_duration_s=max_duration_s)


@AUDIO_LOADER_REGISTRY.register("torchcodec")
class TorchCodecAudioLoader(AudioLoader):
    """Decode via torchcodec, which releases the GIL for the whole decode.

    Recommended when many requests decode audio concurrently; see
    :func:`load_audio_torchcodec`.
    """

    @classmethod
    def load_bytes(
        cls,
        path: BytesIO | Path | str,
        *,
        sr: float | None = None,
        mono: bool = True,
        max_duration_s: float | None = None,
    ) -> tuple[npt.NDArray, float]:
        return load_audio_torchcodec(
            path, sr=sr, mono=mono, max_duration_s=max_duration_s
        )


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

        # Select the decoding backend, e.g.
        #   --media-io-kwargs '{"audio": {"audio_backend": "torchcodec"}}'
        backend = kwargs.pop("audio_backend", None) or envs.VLLM_AUDIO_LOADER_BACKEND
        if backend not in AUDIO_LOADER_REGISTRY.name2class:
            raise ValueError(
                f"Unknown audio_backend {backend!r}. Available backends: "
                f"{sorted(AUDIO_LOADER_REGISTRY.name2class)}"
            )
        if backend == "torchcodec":
            # Fail loudly at construction rather than per request, so a
            # misconfigured deployment is obvious instead of silently
            # falling back.
            check_torchcodec_available()
        self.audio_backend = backend
        self.audio_loader = AUDIO_LOADER_REGISTRY.load(backend)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return self.audio_loader.load_bytes(
            BytesIO(data),
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
        )

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return self.audio_loader.load_bytes(
            filepath,
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
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
