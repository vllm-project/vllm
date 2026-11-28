# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for chunk timestamp offset drift in _preprocess_speech_to_text."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing

SR = 16_000
_PATCH = "vllm.entrypoints.speech_to_text.base.serving"


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

    duration = sum(chunk_lengths) / SR

    expected_offsets = [0.0, 29.5, 29.5 + 29.7]  # cumulative seconds
    wrong_offsets = [0.0, 30.0, 60.0]  # what the old bug produced

    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving._decode_and_chunk_speech_async = AsyncMock(return_value=(chunks, duration))
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
        return_value=[MagicMock()] * len(chunks)
    )

    request = MagicMock()
    request.language = "en"
    request.to_language = None
    request.response_format = "json"
    request.build_stt_params.return_value = MagicMock()

    with patch(f"{_PATCH}.parse_model_prompt", return_value=MagicMock()):
        _, _, offsets = await serving._preprocess_speech_to_text(
            request=request,
            audio_data=b"\x00",
            request_id="test",
        )

    assert offsets == pytest.approx(expected_offsets, abs=1e-6)
    assert offsets != pytest.approx(wrong_offsets, abs=1e-6)


@pytest.mark.asyncio
async def test_language_detection_and_first_chunk_share_audio_uuid():
    """The detection request and the first chunk's request carry the same audio,
    so they must name it with the same UUID for the engine to encode it once.
    """
    chunks = [np.zeros(SR, dtype=np.float32), np.zeros(SR, dtype=np.float32)]

    def encoder_decoder_prompt(decoder_text):
        return {
            "encoder_prompt": {"prompt": "", "multi_modal_data": {"audio": chunks[0]}},
            "decoder_prompt": {"prompt": decoder_text},
        }

    async def detection_output():
        yield MagicMock(finished=True)

    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving._decode_and_chunk_speech_async = AsyncMock(return_value=(chunks, 2.0))
    serving.asr_config = SpeechToTextConfig(sample_rate=float(SR), max_audio_clip_s=30)
    model_cls = MagicMock()
    model_cls.validate_language.side_effect = lambda lang: lang
    model_cls.supports_explicit_language_detection = True
    model_cls.get_language_detection_prompt.side_effect = lambda *_: (
        encoder_decoder_prompt("<|startoftranscript|>")
    )
    model_cls.parse_language_detection_output.return_value = "en"
    model_cls.get_generation_prompt.side_effect = lambda _: encoder_decoder_prompt(
        "<|startoftranscript|><|en|>"
    )
    serving.model_cls = model_cls
    serving.model_config = MagicMock()
    serving.tokenizer = MagicMock()
    serving.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    engine_client = MagicMock()
    engine_client.generate.side_effect = lambda *_, **__: detection_output()
    serving.engine_client = engine_client
    renderer = MagicMock()
    renderer.render_cmpl_async = AsyncMock(return_value=[MagicMock()] * 2)
    serving.renderer = renderer

    request = MagicMock()
    request.language = None
    request.to_language = None
    request.response_format = "json"

    async def preprocess_and_get_prompts():
        with patch(f"{_PATCH}.parse_model_prompt", side_effect=lambda _, p: p):
            await serving._preprocess_speech_to_text(
                request=request,
                audio_data=b"\x00",
                request_id="client-chosen-id",
            )
        request.language = None
        detection_prompt = engine_client.generate.call_args.args[0]
        return detection_prompt, *renderer.render_cmpl_async.call_args.args[0]

    detection_prompt, first_chunk, second_chunk = await preprocess_and_get_prompts()
    detection_uuids = detection_prompt["encoder_prompt"]["multi_modal_uuids"]
    assert first_chunk["encoder_prompt"]["multi_modal_uuids"] == detection_uuids
    assert "multi_modal_uuids" not in second_chunk["encoder_prompt"]

    # A second request reusing the same client request ID must not share audio.
    other_detection_prompt, _, _ = await preprocess_and_get_prompts()
    other_uuids = other_detection_prompt["encoder_prompt"]["multi_modal_uuids"]
    assert other_uuids != detection_uuids
