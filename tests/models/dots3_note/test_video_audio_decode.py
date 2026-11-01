# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from vllm import envs

pytestmark = pytest.mark.cpu_test

_VIDEO_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "models"
    / "dots3_note"
    / "common"
    / "video.py"
)


def _load_video_mod():
    name = "dots3_note_common_video"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _VIDEO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def video_mod():
    return _load_video_mod()


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * max(1, len(text) // 8)


def test_decode_audio_forwards_duration_and_byte_limits(monkeypatch, video_mod):
    captured: dict[str, object] = {}

    def fake_load(
        path,
        *,
        sr=None,
        mono=True,
        max_duration_s=None,
        max_decode_bytes=None,
    ):
        captured["path_type"] = type(path)
        captured["sr"] = sr
        captured["mono"] = mono
        captured["max_duration_s"] = max_duration_s
        captured["max_decode_bytes"] = max_decode_bytes
        return np.zeros(16000, dtype=np.float32), sr

    monkeypatch.setattr("vllm.multimodal.media.audio.load_audio_torchcodec", fake_load)

    pcm, duration = video_mod._decode_audio(b"container-bytes", sample_rate=16000)

    assert captured["path_type"] is BytesIO
    assert captured["sr"] == 16000
    assert captured["mono"] is True
    assert captured["max_duration_s"] == envs.VLLM_MAX_AUDIO_DECODE_DURATION_S
    assert captured["max_decode_bytes"] == envs.VLLM_MAX_AUDIO_DECODE_BYTES
    assert pcm.dtype == np.int16
    assert pcm.shape == (16000,)
    assert duration == pytest.approx(1.0)


def test_decode_audio_rejects_over_duration(monkeypatch, video_mod):
    def fake_load(*_args, **_kwargs):
        raise ValueError(
            "Audio exceeds maximum allowed duration of 600s. Set "
            "VLLM_MAX_AUDIO_DECODE_DURATION_S to increase this limit."
        )

    monkeypatch.setattr("vllm.multimodal.media.audio.load_audio_torchcodec", fake_load)

    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_DURATION_S"):
        video_mod._decode_audio(b"long-audio-container", sample_rate=16000)


def test_decode_audio_rejects_over_decode_bytes(monkeypatch, video_mod):
    def fake_load(*_args, **_kwargs):
        raise ValueError(
            "Audio would allocate 512 MiB of PCM, exceeding the 256 MiB "
            "limit. Set VLLM_MAX_AUDIO_DECODE_BYTES to increase this limit."
        )

    monkeypatch.setattr("vllm.multimodal.media.audio.load_audio_torchcodec", fake_load)

    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        video_mod._decode_audio(b"wide-audio-container", sample_rate=16000)


def test_decode_audio_skips_missing_audio_track(monkeypatch, video_mod):
    def fake_load(*_args, **_kwargs):
        raise ValueError("No audio found in the input.")

    monkeypatch.setattr("vllm.multimodal.media.audio.load_audio_torchcodec", fake_load)

    pcm, duration = video_mod._decode_audio(b"video-without-audio", sample_rate=16000)

    assert pcm is None
    assert duration == 0.0


def test_preprocess_rejects_over_duration_audio(monkeypatch, video_mod):
    def fake_load(*_args, **_kwargs):
        raise ValueError(
            "Audio exceeds maximum allowed duration of 600s. Set "
            "VLLM_MAX_AUDIO_DECODE_DURATION_S to increase this limit."
        )

    monkeypatch.setattr("vllm.multimodal.media.audio.load_audio_torchcodec", fake_load)

    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_DURATION_S"):
        video_mod.preprocess_dots3_note_video(
            b"long-audio-container",
            tokenizer=_DummyTokenizer(),
            question="describe",
            seq=4096,
            audio_cap=1.0,
        )


def test_preprocess_skips_audio_decode_when_audio_cap_is_zero(monkeypatch, video_mod):
    called = False

    def fake_decode(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("audio decode must not run when audio_cap is 0")

    monkeypatch.setattr(video_mod, "_decode_audio", fake_decode)

    def fake_frames(*_args, **_kwargs):
        from PIL import Image

        return [(0.0, Image.new("RGB", (8, 8)))], 1.0

    monkeypatch.setattr(video_mod, "_decode_frames", fake_frames)

    parts = video_mod.preprocess_dots3_note_video(
        b"container-bytes",
        tokenizer=_DummyTokenizer(),
        question="describe",
        seq=4096,
        audio_cap=0.0,
    )

    assert called is False
    assert parts
    assert all(part.kind != "audio" for part in parts)
