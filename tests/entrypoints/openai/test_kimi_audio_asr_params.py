# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins

import numpy as np
import pytest

from vllm.config import SpeechToTextConfig
from vllm.entrypoints.openai.chat_completion.serving import (
    _maybe_add_kimi_stop_tokens as _chat_add_kimi_stop_tokens,
)
from vllm.entrypoints.openai.completion.serving import (
    _maybe_add_kimi_stop_tokens as _completion_add_kimi_stop_tokens,
)
from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText
from vllm.model_executor.models.kimi_audio_asr import KimiAudioForConditionalGeneration
from vllm.sampling_params import SamplingParams


class _DummyTokens:
    def __init__(self):
        self.kimia_text_eos = 151667
        self.msg_end = 151645
        self.media_end = 151663


class _DummyTokenizer:
    def get_stop_token_ids(self):
        return [151645, 151667]


class _DummyHFConfig:
    kimia_token_offset = 1


class _DummyModelConfig:
    hf_config = _DummyHFConfig()


def test_kimia_sampling_params_skips_prefix_cache() -> None:
    sampler = SamplingParams(max_tokens=1, temperature=0.0)
    stt = OpenAISpeechToText.__new__(OpenAISpeechToText)
    stt.__dict__["_kimia_extra_tokens"] = _DummyTokens()

    OpenAISpeechToText._apply_kimia_sampling_params(stt, sampler)

    assert sampler.skip_reading_prefix_cache is True
    assert 151645 in sampler.stop_token_ids
    assert 151667 in sampler.stop_token_ids
    assert 151663 in sampler.stop_token_ids


def test_kimi_stop_tokens_added_for_chat_and_completion() -> None:
    tokenizer = _DummyTokenizer()
    model_config = _DummyModelConfig()
    default_sampling: dict[str, list[int]] = {}

    _chat_add_kimi_stop_tokens(model_config, tokenizer, default_sampling)
    assert set(default_sampling["stop_token_ids"]) == {151645, 151667}

    default_sampling = {"stop_token_ids": [151645]}
    _completion_add_kimi_stop_tokens(model_config, tokenizer, default_sampling)
    assert set(default_sampling["stop_token_ids"]) == {151645, 151667}


def test_kimi_audio_requires_kimia_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("kimia_infer"):
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    audio = np.zeros(16000, dtype=np.float32)
    stt_config = SpeechToTextConfig(sample_rate=16_000, max_audio_clip_s=30)

    with pytest.raises(RuntimeError, match="kimia_infer"):
        KimiAudioForConditionalGeneration.get_generation_prompt(
            audio=audio,
            stt_config=stt_config,
            model_config=object(),
            language="zh",
            task_type="transcribe",
            request_prompt="",
            to_language=None,
        )
