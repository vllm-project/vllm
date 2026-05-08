# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import SpeechToTextConfig
from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
    OpenAISpeechToText,
)
from vllm.sampling_params import SamplingParams


def _make_handler(
    generation_config: dict[str, object],
    tokenizer_eos_token_id: int | None,
) -> OpenAISpeechToText:
    handler = OpenAISpeechToText.__new__(OpenAISpeechToText)
    handler.asr_config = SpeechToTextConfig(generation_config=generation_config)
    handler.renderer = SimpleNamespace(get_eos_token_id=lambda: tokenizer_eos_token_id)
    return handler


def test_openai_transcription_uses_config_eos_when_tokenizer_missing():
    handler = _make_handler({"eos_token_id": 3}, None)

    assert handler._get_generation_config_stop_token_ids() == [3]


def test_openai_transcription_ignores_config_eos_when_tokenizer_matches():
    handler = _make_handler({"eos_token_id": 3}, 3)

    assert handler._get_generation_config_stop_token_ids() == []


def test_openai_transcription_updates_sampling_params_from_generation_config():
    handler = _make_handler({"eos_token_id": 3}, None)

    sampling_params = SamplingParams(max_tokens=8, stop_token_ids=[9])
    handler._update_transcription_sampling_params(sampling_params)

    assert set(sampling_params.stop_token_ids) == {3, 9}
    assert 3 in sampling_params.all_stop_token_ids
    assert 9 in sampling_params.all_stop_token_ids


def test_openai_transcription_uses_tokenizer_eos_as_primary_eos():
    handler = _make_handler({"eos_token_id": 3}, 3)

    sampling_params = SamplingParams(max_tokens=8)
    handler._update_transcription_sampling_params(sampling_params)

    assert sampling_params.eos_token_id == 3
    assert sampling_params.stop_token_ids == []
    assert sampling_params.all_stop_token_ids == {3}


def test_openai_transcription_custom_stop_tokens_reject_beam_search():
    handler = OpenAISpeechToText.__new__(OpenAISpeechToText)
    handler.create_error_response = lambda message: message

    error = handler._get_transcription_beam_search_error([3])

    assert "stop token ids" in error
    assert "beam search" in error
