# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybase64 as base64
import pytest
import soundfile as sf

from vllm.multimodal.media import AudioMediaIO
from vllm.multimodal.media import audio as audio_module
from vllm.multimodal.media.audio import (
    load_audio,
    load_audio_soundfile,
    load_audio_torchcodec,
)

from ...conftest import AudioTestAssets

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


@pytest.fixture
def dummy_audio():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)


@pytest.fixture
def dummy_audio_bytes(audio_assets: AudioTestAssets):
    with open(audio_assets[0].get_local_path(), "rb") as f:
        return f.read()


def test_audio_media_io_load_bytes(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    out = audio_io.load_bytes(dummy_audio_bytes)
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 16000


def test_audio_media_io_load_base64(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    encoded = base64.b64encode(dummy_audio_bytes).decode("utf-8")
    out = audio_io.load_base64("audio/wav", encoded)
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 16000


def test_audio_media_io_load_file(audio_assets: AudioTestAssets):
    audio_io = AudioMediaIO()
    path = audio_assets[0].get_local_path()
    out = audio_io.load_file(path)
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 16000


def test_audio_media_io_encode_base64(dummy_audio):
    audio_io = AudioMediaIO()
    media = (dummy_audio, 16000)
    with patch("soundfile.write") as mock_write:

        def write_to_buffer(buffer, *_args, **_kwargs):
            buffer.write(b"dummy_wav_data")

        mock_write.side_effect = write_to_buffer

        out = audio_io.encode_base64(media)
        decoded = base64.b64decode(out)
        assert decoded == b"dummy_wav_data"
        mock_write.assert_called_once()


def test_load_audio_max_duration_respected(dummy_audio_bytes):
    """Valid audio within the duration limit should load successfully."""
    y, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, max_duration_s=3600)
    assert isinstance(y, np.ndarray)
    assert len(y) > 0


def test_load_audio_max_duration_rejected(dummy_audio_bytes):
    """Audio exceeding the duration limit must be rejected during decode."""
    with pytest.raises(ValueError, match="exceeds maximum allowed duration"):
        load_audio(BytesIO(dummy_audio_bytes), sr=None, max_duration_s=0.0001)


def test_audio_media_io_from_video(video_assets):
    audio_io = AudioMediaIO()
    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        audio, sr = audio_io.load_bytes(f.read())
    audio_ref, sr_ref = load_audio(video_path, sr=None)
    assert sr == sr_ref
    np.testing.assert_allclose(audio_ref, audio, atol=1e-4)


def _make_flac_bytes(frames: int, channels: int, samplerate: int) -> bytes:
    """Create a minimal FLAC file in memory for testing."""
    data = np.zeros((frames, channels), dtype=np.int16)
    buf = BytesIO()
    sf.write(buf, data, samplerate, format="FLAC")
    return buf.getvalue()


def test_small_file_passes_memory_guard():
    """A small valid file should pass both duration and memory guards."""
    payload = _make_flac_bytes(frames=16000, channels=1, samplerate=16000)
    y, sr = load_audio_soundfile(
        BytesIO(payload),
        sr=None,
        max_duration_s=600,
        max_decode_bytes=256 * 1024 * 1024,
    )
    assert isinstance(y, np.ndarray)
    assert len(y) == 16000


def test_memory_guard_rejects_large_allocation():
    """A file whose frames*channels*4 exceeds the byte limit must be
    rejected before allocating the buffer."""
    # 100_000 frames * 8 channels * 4 bytes = 3.2 MB
    payload = _make_flac_bytes(frames=100_000, channels=8, samplerate=48000)
    # Set limit to 1 MiB — should reject
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_soundfile(
            BytesIO(payload),
            sr=None,
            max_duration_s=600,
            max_decode_bytes=1 * 1024 * 1024,
        )


def test_forged_samplerate_rejected_by_memory_guard():
    """The PoC scenario: high sample rate fools the duration guard but
    the memory guard catches the large frame*channel allocation."""
    # Forged high sample rate: 655350 Hz, 8 channels, 1M frames
    # Duration guard sees: 1_000_000 / 655_350 = 1.5s → passes
    # Memory: 1_000_000 * 8 * 4 = 32 MB
    payload = _make_flac_bytes(frames=1_000_000, channels=8, samplerate=655350)
    # Set memory limit to 16 MiB — below the 32 MB allocation
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_soundfile(
            BytesIO(payload),
            sr=None,
            max_duration_s=600,
            max_decode_bytes=16 * 1024 * 1024,
        )


def test_load_audio_threads_max_decode_bytes():
    """Verify load_audio passes max_decode_bytes through to backend."""
    # 50_000 frames * 4 channels * 4 bytes = 800 KB
    payload = _make_flac_bytes(frames=50_000, channels=4, samplerate=44100)
    # Limit of 512 KB should reject
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio(
            BytesIO(payload),
            sr=None,
            max_duration_s=600,
            max_decode_bytes=512 * 1024,
        )


@pytest.mark.parametrize("backend", ["soundfile", "pyav", "torchcodec"])
def test_load_audio_backend_matches_default(backend, dummy_audio_bytes):
    """Every explicit backend must decode to the same samples as `auto`."""
    if backend == "torchcodec":
        pytest.importorskip("torchcodec")
    ref_audio, ref_sr = load_audio(BytesIO(dummy_audio_bytes), sr=None)
    audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, backend=backend)
    assert sr == ref_sr
    # Decoders disagree only on codec encoder-delay/padding, so torchcodec may
    # emit a few extra trailing samples (e.g. ~192 for Ogg Vorbis). Compare the
    # overlapping region, which must agree to float32 precision.
    n = min(ref_audio.shape[-1], audio.shape[-1])
    assert n > 0
    np.testing.assert_allclose(ref_audio[:n], audio[:n], atol=1e-4)


def test_load_audio_unknown_backend_rejected(dummy_audio_bytes):
    """An unknown backend must fail loudly instead of silently degrading."""
    with pytest.raises(ValueError, match="Unknown audio backend"):
        load_audio(BytesIO(dummy_audio_bytes), sr=None, backend="not_a_backend")


def test_load_audio_auto_falls_back_without_torchcodec(dummy_audio_bytes):
    """`auto` must fall back to the soundfile → PyAV chain when torchcodec
    is not importable."""
    ref_audio, ref_sr = load_audio_soundfile(BytesIO(dummy_audio_bytes), sr=None)
    with patch.object(audio_module, "load_audio_torchcodec", side_effect=ImportError):
        audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, backend="auto")
    assert sr == ref_sr
    np.testing.assert_array_equal(ref_audio, audio)


def test_audio_media_io_audio_backend_kwarg(dummy_audio_bytes):
    """`audio_backend` selects the backend; unknown values fail at init."""
    audio, sr = AudioMediaIO(audio_backend="pyav").load_bytes(dummy_audio_bytes)
    assert isinstance(audio, np.ndarray)
    assert sr == 16000
    with pytest.raises(ValueError, match="Unknown audio_backend"):
        AudioMediaIO(audio_backend="not_a_backend")


def test_torchcodec_max_duration_rejected(dummy_audio_bytes):
    """The decompression-bomb duration guard must hold for torchcodec too."""
    pytest.importorskip("torchcodec")
    with pytest.raises(ValueError, match="exceeds maximum allowed duration"):
        load_audio_torchcodec(
            BytesIO(dummy_audio_bytes), sr=None, max_duration_s=0.0001
        )


def test_torchcodec_max_decode_bytes_rejected(dummy_audio_bytes):
    """The decompression-bomb memory guard must hold for torchcodec too."""
    pytest.importorskip("torchcodec")
    audio, _ = load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None)
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_torchcodec(
            BytesIO(dummy_audio_bytes),
            sr=None,
            max_decode_bytes=audio.nbytes - 1,
        )
