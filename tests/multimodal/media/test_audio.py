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
from vllm.multimodal.media.audio import load_audio, load_audio_soundfile

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
