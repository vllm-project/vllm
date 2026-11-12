# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request priority reaches the engine for every speech-to-text engine call."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import Request

from tests.entrypoints.speech_to_text.transcription.test_transcription_inter_chunk_spacing import (  # noqa: E501
    _request_output,
    _StubTranscriptionModel,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.entrypoints.speech_to_text.translation.serving import (
    OpenAIServingTranslation,
)
from vllm.exceptions import VLLMValidationError

_ENDPOINTS: dict[
    str,
    tuple[
        type[OpenAIServingTranscription] | type[OpenAIServingTranslation],
        type[TranscriptionRequest] | type[TranslationRequest],
        str,
    ],
] = {
    "transcription": (
        OpenAIServingTranscription,
        TranscriptionRequest,
        "create_transcription",
    ),
    "translation": (OpenAIServingTranslation, TranslationRequest, "create_translation"),
}


def _engine_client(priorities: list[int]) -> MagicMock:
    async def generate(*args, priority=0, **kwargs):
        priorities.append(priority)
        yield _request_output("hello")

    engine_client = MagicMock()
    engine_client.model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config.max_model_len = 8192
    engine_client.errored = False
    engine_client.is_tracing_enabled = AsyncMock(return_value=False)
    engine_client.generate = generate
    return engine_client


def _raw_request(header: str | None) -> Request:
    headers = [] if header is None else [(b"x-vllm-priority", header.encode())]
    return Request({"type": "http", "headers": headers})


async def _create(endpoint: str, priorities: list[int], header: str | None):
    serving_cls, request_cls, create = _ENDPOINTS[endpoint]
    models = MagicMock(spec=OpenAIServingModels)
    models.lora_requests = {}
    models.is_base_model.return_value = True
    preprocess = AsyncMock(return_value=([MagicMock(), MagicMock()], 6.0, [0.0, 5.0]))

    with (
        patch(
            "vllm.model_executor.model_loader.get_model_cls",
            return_value=_StubTranscriptionModel,
        ),
        patch.object(SpeechToTextBaseServing, "_preprocess_speech_to_text", preprocess),
    ):
        serving = serving_cls(_engine_client(priorities), models, request_logger=None)
        request = request_cls.model_construct(
            file=MagicMock(),
            model="stub-model",
            language="en",
            stream=False,
            response_format="json",
            priority=-9,
        )
        await getattr(serving, create)(b"\x00\x00", request, _raw_request(header))
    return preprocess


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
@pytest.mark.parametrize("header", [None, "-5", "0"])
async def test_priority_reaches_every_chunk(endpoint, header):
    """Body priority, or the header that overrides it, is used for each chunk."""
    priorities: list[int] = []
    preprocess = await _create(endpoint, priorities, header)

    expected = -9 if header is None else int(header)
    assert priorities == [expected, expected]
    assert preprocess.await_args.kwargs["priority"] == expected


@pytest.mark.asyncio
async def test_out_of_range_priority_header_is_rejected():
    with pytest.raises(VLLMValidationError):
        await _create("transcription", [], str(2**31))


@pytest.mark.asyncio
async def test_language_detection_uses_request_priority():
    priorities: list[int] = []
    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server.engine_client = _engine_client(priorities)
    server.asr_config = SimpleNamespace()
    server.tokenizer = Mock()
    server.model_cls = SimpleNamespace(
        get_language_detection_prompt=Mock(return_value={"prompt": "detect"}),
        get_language_token_ids=Mock(return_value=[1]),
        parse_language_detection_output=Mock(return_value="en"),
    )

    assert await server._detect_language(Mock(), "lang-detect", priority=-3) == "en"
    assert priorities == [-3]
