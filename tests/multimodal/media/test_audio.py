# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybase64 as base64
import pytest
from io import BytesIO

import av

from vllm.multimodal.media import AudioMediaIO
from vllm.multimodal.media.audio import load_audio

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


@pytest.fixture(params=[False, True], ids=["with-metadata", "cueless"])
def dummy_webm_bytes(request):


    rate = 48000
    total = rate  # 1 second
    t = np.arange(total, dtype=np.float32) / rate
    pcm = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    options = {"live": "1"} if request.param else None
    buf = BytesIO()
    with av.open(buf, "w", format="webm", options=options) as container:
        stream = container.add_stream("libopus", rate=rate)
        stream.layout = "mono"
        frame_size = stream.codec_context.frame_size or 960
        for offset in range(0, total, frame_size):
            chunk = pcm[offset : offset + frame_size]
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), format="flt", layout="mono"
            )
            frame.sample_rate = rate
            frame.pts = offset
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


def test_load_audio_pyav_fallback_within_duration(dummy_webm_bytes):
    """A WebM under the duration limit should load via the PyAV fallback."""

    y, sr = load_audio(BytesIO(dummy_webm_bytes), sr=None, max_duration_s=3600)
    assert isinstance(y, np.ndarray)
    assert len(y) > 0


def test_load_audio_pyav_fallback_max_duration_rejected(dummy_webm_bytes):
    """The duration guard's message must survive the PyAV fallback."""

    with pytest.raises(ValueError, match="exceeds maximum allowed duration"):
        load_audio(BytesIO(dummy_webm_bytes), sr=None, max_duration_s=0.25)


def test_load_audio_invalid_bytes_rejected():
    """Undecodable bytes keep the generic invalid-file error."""

    with pytest.raises(ValueError, match="Invalid or unsupported audio file"):
        load_audio(BytesIO(b"\x00\x01not-audio"), sr=None, max_duration_s=3600)


def test_audio_media_io_from_video(video_assets):
    audio_io = AudioMediaIO()
    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        audio, sr = audio_io.load_bytes(f.read())
    audio_ref, sr_ref = load_audio(video_path, sr=None)
    assert sr == sr_ref
    np.testing.assert_allclose(audio_ref, audio, atol=1e-4)
