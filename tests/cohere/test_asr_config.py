# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

import vllm.envs as envs
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText

# https://github.com/vllm-project/vllm/pull/44612#discussion_r3360362582
DEFAULT_NUM_WORKERS = 2


class _StubTranscriptionModel:
    supports_segment_timestamp = False

    @classmethod
    def get_speech_to_text_config(cls, model_config, task_type):
        return SpeechToTextConfig(
            sample_rate=16000.0,
            max_audio_clip_s=5.0,
        )


def test_audio_preprocess_workers_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_MAX_AUDIO_PREPROCESS_WORKERS", raising=False)
    monkeypatch.setattr(envs.os, "cpu_count", lambda: 64)

    cache_clear = getattr(getattr(envs, "__getattr__", None), "cache_clear", None)
    if cache_clear is not None:
        cache_clear()

    assert envs.VLLM_MAX_AUDIO_PREPROCESS_WORKERS == DEFAULT_NUM_WORKERS


def test_speech_to_text_preprocess_executor_num_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pick the default value to test instead of patching the environment variable
    # monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_PREPROCESS_WORKERS", 2)

    engine_client = MagicMock()
    engine_client.model_config = MagicMock()
    engine_client.model_config.get_diff_sampling_param.return_value = {}

    models = MagicMock(spec=OpenAIServingModels)
    models.lora_requests = {}
    models.is_base_model.return_value = True

    executor_mock = MagicMock()
    async_wrapper_mock = MagicMock()

    with (
        patch(
            "vllm.model_executor.model_loader.get_model_cls",
            return_value=_StubTranscriptionModel,
        ),
        patch(
            "vllm.entrypoints.openai.speech_to_text.speech_to_text.ThreadPoolExecutor",
            return_value=executor_mock,
        ) as executor_cls,
        patch(
            "vllm.entrypoints.openai.speech_to_text.speech_to_text.make_async_with_semaphore",
            return_value=async_wrapper_mock,
        ) as make_async_mock,
    ):
        serving = OpenAISpeechToText(
            engine_client,
            models,
            request_logger=None,
        )

    executor_cls.assert_called_once_with(
        max_workers=DEFAULT_NUM_WORKERS,
        thread_name_prefix="stt-preprocess",
    )
    assert serving._preprocess_executor is executor_mock
    make_async_mock.assert_called_once_with(
        serving._decode_and_chunk_speech,
        executor=executor_mock,
    )
    assert serving._decode_and_chunk_speech_async is async_wrapper_mock
