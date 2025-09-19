# test_audio.py
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vllm.multimodal.audio import (
    resample_audio_librosa,
    resample_audio_scipy,
    AudioResampler,
    AudioMediaIO,
)


@pytest.fixture
def dummy_audio():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)


def test_resample_audio_librosa(dummy_audio):
    with patch("vllm.multimodal.audio.librosa.resample") as mock_resample:
        mock_resample.return_value = dummy_audio * 2
        out = resample_audio_librosa(dummy_audio, orig_sr=44100, target_sr=22050)
        mock_resample.assert_called_once_with(dummy_audio, orig_sr=44100, target_sr=22050)
        assert np.all(out == dummy_audio * 2)


def test_resample_audio_scipy(dummy_audio):
    out_down = resample_audio_scipy(dummy_audio, orig_sr=4, target_sr=2)
    out_up = resample_audio_scipy(dummy_audio, orig_sr=2, target_sr=4)
    out_same = resample_audio_scipy(dummy_audio, orig_sr=4, target_sr=4)

    assert len(out_down) == 3  
    assert len(out_up) == 10  
    assert np.all(out_same == dummy_audio)
    
    out_non_int = resample_audio_scipy(dummy_audio, orig_sr=3, target_sr=2)
    assert len(out_non_int) == 4


def test_audio_resampler_librosa_calls_resample(dummy_audio):
    resampler = AudioResampler(target_sr=22050, method="librosa")
    with patch("vllm.multimodal.audio.resample_audio_librosa") as mock_resample:
        mock_resample.return_value = dummy_audio
        out = resampler.resample(dummy_audio, orig_sr=44100)
        mock_resample.assert_called_once_with(dummy_audio, orig_sr=44100, target_sr=22050)
        assert np.all(out == dummy_audio)


def test_audio_resampler_scipy_calls_resample(dummy_audio):
    resampler = AudioResampler(target_sr=22050, method="scipy")
    with patch("vllm.multimodal.audio.resample_audio_scipy") as mock_resample:
        mock_resample.return_value = dummy_audio
        out = resampler.resample(dummy_audio, orig_sr=44100)
        mock_resample.assert_called_once_with(dummy_audio, orig_sr=44100, target_sr=22050)
        assert np.all(out == dummy_audio)


def test_audio_resampler_invalid_method(dummy_audio):
    resampler = AudioResampler(target_sr=22050, method="invalid")
    with pytest.raises(ValueError):
        resampler.resample(dummy_audio, orig_sr=44100)


def test_audio_resampler_no_target_sr(dummy_audio):
    resampler = AudioResampler(target_sr=None)
    with pytest.raises(RuntimeError):
        resampler.resample(dummy_audio, orig_sr=44100)


@pytest.fixture
def dummy_audio_bytes():
    return b"FAKEAUDIOBYTES"


def test_audio_media_io_load_bytes(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    with patch("vllm.multimodal.audio.librosa.load") as mock_load:
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
    with patch("vllm.multimodal.audio.librosa.load") as mock_load:
        mock_load.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_file(path)
        mock_load.assert_called_once_with(path, sr=None)
        assert isinstance(out[0], np.ndarray)
        assert out[1] == 16000


def test_audio_media_io_encode_base64(dummy_audio):
    audio_io = AudioMediaIO()
    media = (dummy_audio, 16000)
    with patch("vllm.multimodal.audio.soundfile.write") as mock_write:  
        def write_to_buffer(buffer, *_args, **_kwargs):  
            buffer.write(b"dummy_wav_data")  
        mock_write.side_effect = write_to_buffer  

        out = audio_io.encode_base64(media)  
        decoded = base64.b64decode(out)  
        assert decoded == b"dummy_wav_data"  
        mock_write.assert_called_once()  
