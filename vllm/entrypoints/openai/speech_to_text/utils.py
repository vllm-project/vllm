# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio decoding utilities for the speech-to-text endpoints."""

import io
import os
import subprocess

import numpy as np

from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]

logger = init_logger(__name__)

# Public libsndfile error codes exposed via ``soundfile.LibsndfileError.code``.
# soundfile is librosa's primary backend.  These codes indicate that the audio
# data itself is problematic (unrecognised container, corrupt file, or
# unsupported encoding) rather than a transient server error.
# 1 = unrecognised format, 3 = malformed file, 4 = unsupported encoding
_BAD_SF_CODES = {1, 3, 4}


def _decode_audio_bytes_ffmpeg(
    audio_data: bytes,
    sr: int | float,
) -> tuple[np.ndarray, int]:
    """
    Decode audio bytes to float32 PCM via ffmpeg, entirely in memory.

    Uses ``os.memfd_create`` to present the raw bytes to ffmpeg as a
    seekable file descriptor (via ``/proc/self/fd/<N>``).  This avoids
    writing to disk (tempfile) while still allowing ffmpeg to seek, which is
    required for container formats like MP4/M4A whose metadata (the
    ``moov`` atom) may be located at the end of the file.

    The output is mono float32 audio resampled to *sr* Hz, matching the
    return convention of ``librosa.load``.
    """
    sr = int(sr)
    fd = os.memfd_create("vllm_audio")
    try:
        os.write(fd, audio_data)
        os.lseek(fd, 0, os.SEEK_SET)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            f"/proc/self/fd/{fd}",
            "-vn",  # discard video
            "-ac",
            "1",  # mono
            "-ar",
            str(sr),  # target sample rate
            "-f",
            "f32le",  # raw float32 little-endian PCM
            "pipe:1",  # write to stdout
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            pass_fds=(fd,),  # inherit only this fd
        )
    finally:
        os.close(fd)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to decode audio: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )

    y = np.frombuffer(result.stdout, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError(
            "ffmpeg produced no audio samples (file may be empty or corrupt)"
        )
    return y, sr


def load_audio_bytes(
    audio_data: bytes,
    sr: int | float,
) -> tuple[np.ndarray, int]:
    """Load audio from raw bytes, with an in-memory ffmpeg fallback.

    First tries ``librosa.load(BytesIO(...))`` which works for formats
    that *soundfile* can auto-detect (WAV, FLAC, MP3, OGG, ...).  If
    that fails with a ``LibsndfileError`` indicating an unrecognised or
    unsupported format (typically container formats like MP4/M4A/WebM),
    the bytes are decoded via ffmpeg using an in-memory file descriptor
    so that ffmpeg can seek the container metadata without any disk I/O.
    """
    # default: should work for most formats
    try:
        with io.BytesIO(audio_data) as buf:
            return librosa.load(buf, sr=sr)  # type: ignore[return-value]
    except Exception as exc:
        # Only fall back for known soundfile format-detection failures.
        # Re-raise anything else (e.g. OOM, keyboard interrupt).
        if (
            sf is not None
            and isinstance(exc, sf.LibsndfileError)
            and exc.code in _BAD_SF_CODES
        ):
            logger.debug(
                "librosa/soundfile could not decode audio from BytesIO "
                "(code=%s: %s); falling back to ffmpeg in-memory decode",
                exc.code,
                exc,
            )
        else:
            raise

    # fallback: ffmpeg via in-memory fd
    return _decode_audio_bytes_ffmpeg(audio_data, sr)
