# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText
from vllm.model_executor.models.parakeet_tdt import ParakeetForTDT
from vllm.sampling_params import SamplingParams


class _FakeTranscriptionModel:
    @classmethod
    def get_transcription_stop_token_ids(cls, model_config):
        return [3]


def test_parakeet_transcription_stop_token_ids():
    model_config = SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=3))

    assert ParakeetForTDT.get_transcription_stop_token_ids(model_config) == [3]


def test_openai_transcription_merges_model_stop_token_ids():
    handler = OpenAISpeechToText.__new__(OpenAISpeechToText)
    handler.model_cls = _FakeTranscriptionModel
    handler.model_config = SimpleNamespace()

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
