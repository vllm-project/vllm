# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio decoding utilities for the speech-to-text endpoints."""

import io

import numpy as np
import torchaudio

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


def _decode_audio_bytes_torchaudio(
    audio_data: bytes,
    sr: int,
) -> tuple[np.ndarray, int]:
    """Decode audio bytes to mono float32 PCM via torchaudio, in-process.

    ``torchaudio.load`` (backed by TorchCodec / FFmpeg) can decode
    container formats (MP4, M4A, WebM) directly from a ``BytesIO``
    buffer without spawning a subprocess.  The decoded waveform is
    down-mixed to mono and resampled to *sr* Hz, matching the return
    convention of ``librosa.load``.
    """
    buf = io.BytesIO(audio_data)
    waveform, orig_sr = torchaudio.load(buf)

    # Down-mix to mono (average across channels).
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to the target sample rate when necessary.
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=orig_sr, new_freq=sr
        )

    # Squeeze channel dim → 1-D float32 numpy array (same as librosa.load).
    y = waveform.squeeze(0).numpy()
    if y.size == 0:
        raise RuntimeError(
            "torchaudio produced no audio samples (file may be empty or corrupt)"
        )
    return y, sr


def load_audio_bytes(
    audio_data: bytes,
    sr: int | float,
) -> tuple[np.ndarray, int]:
    """Load audio from raw bytes, with an in-process torchaudio fallback.

    First tries ``librosa.load(BytesIO(...))`` which works for formats
    that *soundfile* can auto-detect (WAV, FLAC, MP3, OGG, ...).  If
    that fails with a ``LibsndfileError`` indicating an unrecognised or
    unsupported format (typically container formats like MP4/M4A/WebM),
    the bytes are decoded in-process via ``torchaudio`` (backed by
    TorchCodec / FFmpeg) which handles these containers natively.
    """
    sr = int(sr)

    # default: soundfile (librosa)
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
                "(code=%s: %s); falling back to torchaudio in-process decode",
                exc.code,
                exc,
            )
        else:
            raise

    # Fallback: torchaudio in-process decode (no subprocess overhead).
    return _decode_audio_bytes_torchaudio(audio_data, sr)
