# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for chunk timestamp offsets in _preprocess_speech_to_text."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig, VADConfig
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.multimodal.audio import VAD

SR = 16_000
_PATCH = "vllm.entrypoints.speech_to_text.base.serving"


def _make_serving(audio: np.ndarray, vad: VAD | None = None):
    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving._decode_audio = MagicMock(  # type: ignore[method-assign]
        return_value=(audio, SR, audio.shape[-1] / SR)
    )
    serving._decode_and_chunk_speech_async = AsyncMock(
        side_effect=serving._decode_and_chunk_speech
    )
    serving._vad_executor = None
    serving._vad_provider = MagicMock()
    serving._vad_provider.get.return_value = vad
    serving.asr_config = SpeechToTextConfig(
        sample_rate=float(SR),
        max_audio_clip_s=30,
        overlap_chunk_second=1,
        min_energy_split_window_size=1600,
    )
    model_cls = MagicMock()
    model_cls.validate_language.side_effect = lambda lang: lang
    model_cls.supports_explicit_language_detection = False
    model_cls.get_generation_prompt.return_value = {}
    serving.model_cls = model_cls
    serving.model_config = MagicMock()
    serving.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    serving.renderer = MagicMock()
    serving.renderer.render_cmpl_async = AsyncMock(
        side_effect=lambda prompts: [MagicMock() for _ in prompts]
    )
    return serving


def _make_request(vad_enabled: bool):
    request = MagicMock()
    request.language = "en"
    request.to_language = None
    request.response_format = "json"
    request.build_stt_params.return_value = MagicMock()
    request.build_vad_config.return_value = VADConfig(enabled=vad_enabled)
    return request


async def _preprocess(serving, request):
    with patch(f"{_PATCH}.parse_model_prompt", return_value=MagicMock()):
        return await serving._preprocess_speech_to_text(
            request=request,
            audio_data=b"\x00",
            request_id="test",
        )


@pytest.mark.asyncio
async def test_chunk_offsets_are_cumulative_not_nominal():
    """chunk_start_offsets must be cumulative actual chunk lengths, not the old approach
    of 'idx * max_audio_clip_s'.  When split_audio places a boundary before the
    nominal 30 s mark, the old formula drifts; the fixed formula stays exact.
    """
    # Chunks shorter than exactly 30 s, as split_audio produces when a quiet
    # region falls inside the 1 s overlap window before the nominal boundary.
    chunk_lengths = [int(29.5 * SR), int(29.7 * SR), int(5.0 * SR)]
    chunks = [np.zeros(n, dtype=np.float32) for n in chunk_lengths]
    audio = np.concatenate(chunks)

    expected_offsets = [0.0, 29.5, 29.5 + 29.7]  # cumulative seconds
    wrong_offsets = [0.0, 30.0, 60.0]  # what the old bug produced

    serving = _make_serving(audio)
    with patch("vllm.multimodal.audio.split_audio", return_value=chunks):
        _, _, offsets = await _preprocess(serving, _make_request(vad_enabled=False))

    assert offsets == pytest.approx(expected_offsets, abs=1e-6)
    assert offsets != pytest.approx(wrong_offsets, abs=1e-6)


@pytest.mark.asyncio
async def test_vad_chunk_offsets_follow_original_audio():
    """With VAD, non-speech between chunks is dropped, so chunk offsets must be
    the speech start times in the original audio, not cumulative chunk lengths.
    Otherwise segment timestamps drift earlier after every removed silence.
    """
    audio = np.random.default_rng(0).standard_normal(100 * SR).astype(np.float32)
    # The second span is longer than max_audio_clip_s, so it is RMS-split.
    speech_spans_s = [(10.0, 20.0), (45.0, 85.0)]
    vad = MagicMock(spec=VAD)
    vad.get_speech_timestamps.return_value = [
        {"start": int(start * SR), "end": int(end * SR)}
        for start, end in speech_spans_s
    ]

    serving = _make_serving(audio, vad=vad)
    request = _make_request(vad_enabled=True)
    _, _, offsets = await _preprocess(serving, request)

    chunk_audios = [
        call.kwargs["audio"] for call in request.build_stt_params.call_args_list
    ]
    assert len(offsets) == len(chunk_audios) == 3
    assert offsets[0] == pytest.approx(10.0)
    assert offsets[1] == pytest.approx(45.0)
    assert offsets[2] == pytest.approx(45.0 + chunk_audios[1].shape[-1] / SR)
    # Each chunk sent to the model is the original audio at its offset.
    for offset, chunk_audio in zip(offsets, chunk_audios):
        start = round(offset * SR)
        np.testing.assert_array_equal(
            chunk_audio, audio[start : start + chunk_audio.shape[-1]]
        )
