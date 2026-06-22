# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for chunk timestamp offset drift in _preprocess_speech_to_text."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.speech_to_text.base.serving import OpenAISpeechToText

SR = 16_000
_PATCH = "vllm.entrypoints.speech_to_text.base.serving"


@pytest.mark.asyncio
async def test_chunk_offsets_are_cumulative_not_nominal():
    """
    chunk_start_offsets must be cumulative actual chunk lengths, not the old approach of
    'idx * max_audio_clip_s'.  When split_audio places a boundary before the
    nominal 30 s mark, the old formula drifts; the fixed formula stays exact.
    """
    # Chunks shorter than exactly 30 s, as split_audio produces when a quiet
    # region falls inside the 1 s overlap window before the nominal boundary.
    chunk_lengths = [int(29.5 * SR), int(29.7 * SR), int(5.0 * SR)]
    chunks = [np.zeros(n, dtype=np.float32) for n in chunk_lengths]

    duration = sum(chunk_lengths) / SR

    expected_offsets = [0.0, 29.5, 29.5 + 29.7]  # cumulative seconds
    wrong_offsets = [0.0, 30.0, 60.0]  # what the old bug produced

    serving = OpenAISpeechToText.__new__(OpenAISpeechToText)
    serving._decode_and_chunk_speech_async = AsyncMock(return_value=(chunks, duration))
    serving.asr_config = SpeechToTextConfig(
        sample_rate=float(SR),
        max_audio_clip_s=30,
        overlap_chunk_second=1,
        min_energy_split_window_size=1600,
    )
    serving.max_audio_filesize_mb = 100.0
    serving.model_cls = MagicMock()
    serving.model_cls.validate_language.side_effect = lambda lang: lang
    serving.model_cls.supports_explicit_language_detection = False
    serving.model_cls.get_generation_prompt.return_value = {}
    serving.model_config = MagicMock()
    serving.task_type = "transcribe"
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
