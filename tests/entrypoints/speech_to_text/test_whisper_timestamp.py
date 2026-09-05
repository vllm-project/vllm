# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.whisper import WHISPER_TIMESTAMP_RULES_KEY


class _WhisperTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens):
        assert text == "<|notimestamps|>"
        assert not add_special_tokens
        return [5]


def test_verbose_prompt_enters_timestamp_mode_without_forcing_zero_timestamp():
    serving = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="whisper"))
    )
    prompt = {
        "encoder_prompt": {"prompt": ""},
        "decoder_prompt": {
            "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        },
    }

    processed = SpeechToTextBaseServing._preprocess_verbose_prompt(serving, prompt)

    decoder_prompt = processed["decoder_prompt"]["prompt"]
    assert "<|notimestamps|>" not in decoder_prompt
    assert not decoder_prompt.endswith("<|0.00|>")


def test_verbose_request_enables_timestamp_rules():
    serving = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="whisper")),
        tokenizer=_WhisperTokenizer(),
    )
    sampling_params = SamplingParams(extra_args={"preserved": True})

    SpeechToTextBaseServing._enable_whisper_timestamp_rules(
        serving, sampling_params, begin_index=4
    )

    assert sampling_params.extra_args["preserved"] is True
    assert sampling_params.extra_args[WHISPER_TIMESTAMP_RULES_KEY] == {
        "eos_token_id": 2,
        "no_timestamps_token_id": 5,
        "max_initial_timestamp_index": 50,
        "begin_index": 4,
    }
