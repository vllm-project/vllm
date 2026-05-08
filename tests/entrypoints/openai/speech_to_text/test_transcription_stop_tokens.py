# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.openai.speech_to_text import speech_to_text as stt_module
from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
    OpenAISpeechToText,
)
from vllm.sampling_params import SamplingParams


def _make_handler(config_eos_token_id: int | None) -> OpenAISpeechToText:
    handler = OpenAISpeechToText.__new__(OpenAISpeechToText)
    handler.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(eos_token_id=config_eos_token_id)
    )
    return handler


def test_openai_transcription_uses_config_eos_when_tokenizer_missing(monkeypatch):
    handler = _make_handler(3)
    monkeypatch.setattr(
        stt_module,
        "cached_tokenizer_from_config",
        lambda model_config: SimpleNamespace(eos_token_id=None),
    )

    assert handler._get_transcription_stop_token_ids() == [3]


def test_openai_transcription_ignores_config_eos_when_tokenizer_matches(monkeypatch):
    handler = _make_handler(3)
    monkeypatch.setattr(
        stt_module,
        "cached_tokenizer_from_config",
        lambda model_config: SimpleNamespace(eos_token_id=3),
    )

    assert handler._get_transcription_stop_token_ids() == []


def test_openai_transcription_merges_config_eos_stop_token(monkeypatch):
    handler = _make_handler(3)
    monkeypatch.setattr(
        stt_module,
        "cached_tokenizer_from_config",
        lambda model_config: SimpleNamespace(eos_token_id=None),
    )

    sampling_params = SamplingParams(max_tokens=8, stop_token_ids=[9])
    stop_token_ids = handler._get_transcription_stop_token_ids()
    handler._apply_transcription_stop_token_ids(sampling_params, stop_token_ids)

    assert sampling_params.stop_token_ids == [3, 9]
    assert 3 in sampling_params.all_stop_token_ids
    assert 9 in sampling_params.all_stop_token_ids


def test_openai_transcription_custom_stop_tokens_reject_beam_search():
    handler = OpenAISpeechToText.__new__(OpenAISpeechToText)
    handler.create_error_response = lambda message: message

    error = handler._get_transcription_beam_search_error([3])

    assert "stop token ids" in error
    assert "beam search" in error
