# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import io

import numpy as np
import pytest
from fastapi import UploadFile

from vllm.config import SpeechToTextConfig
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest
from vllm.model_executor.models.kimi_audio_asr import KimiAudioForConditionalGeneration


def test_kimi_audio_stt_config_defaults() -> None:
    stt_config = KimiAudioForConditionalGeneration.get_speech_to_text_config(
        model_config=object(),
        task_type="transcribe",
    )

    assert stt_config.sample_rate == 16_000
    assert stt_config.max_audio_clip_s == 30
    assert stt_config.skip_reading_prefix_cache is True
    assert stt_config.default_sampling_params == {
        "temperature": 0.0,
        "top_k": 5,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }


def test_transcription_request_uses_stt_defaults() -> None:
    stt_config = KimiAudioForConditionalGeneration.get_speech_to_text_config(
        model_config=object(),
        task_type="transcribe",
    )
    dummy_file = UploadFile(filename="dummy.wav", file=io.BytesIO(b""))
    request = TranscriptionRequest(file=dummy_file)

    sampling_params = request.to_sampling_params(
        default_max_tokens=16,
        default_sampling_params=stt_config.default_sampling_params,
    )
    if stt_config.skip_reading_prefix_cache:
        sampling_params.skip_reading_prefix_cache = True

    assert sampling_params.temperature == 0.0
    assert sampling_params.top_k == 5
    assert sampling_params.top_p == 1.0
    assert sampling_params.min_p == 0.0
    assert sampling_params.repetition_penalty == 1.0
    assert sampling_params.skip_reading_prefix_cache is True


def test_entrypoints_no_kimi_stop_token_helpers() -> None:
    import vllm.entrypoints.openai.chat_completion.serving as chat_serving
    import vllm.entrypoints.openai.completion.serving as completion_serving

    assert not hasattr(chat_serving, "_maybe_add_kimi_stop_tokens")
    assert not hasattr(completion_serving, "_maybe_add_kimi_stop_tokens")


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
