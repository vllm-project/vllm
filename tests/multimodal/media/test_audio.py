# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vllm.multimodal.media import AudioMediaIO

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


@pytest.fixture
def dummy_audio():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)


@pytest.fixture
def dummy_audio_bytes():
    return b"FAKEAUDIOBYTES"


def test_audio_media_io_load_bytes(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    with patch("librosa.load") as mock_load:
        mock_load.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_bytes(dummy_audio_bytes)
        mock_load.assert_called_once()
        assert isinstance(out[0], np.ndarray)
        assert out[1] == 16000


def test_audio_media_io_load_base64(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    encoded = base64.b64encode(dummy_audio_bytes).decode("utf-8")
    with patch.object(AudioMediaIO, "load_bytes") as mock_load_bytes:
        mock_load_bytes.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_base64("audio/wav", encoded)
        mock_load_bytes.assert_called_once()
        assert isinstance(out[0], np.ndarray)
        assert out[1] == 16000


def test_audio_media_io_load_file():
    audio_io = AudioMediaIO()
    path = Path("/fake/path.wav")
    with patch("librosa.load") as mock_load:
        mock_load.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_file(path)
        mock_load.assert_called_once_with(path, sr=None)
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
