# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.realtime.api_router import realtime_endpoint
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.serving import (
    OpenAIServingTranslation,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("serving_class", "method_name"),
    [
        (OpenAIServingTranscription, "create_transcription"),
        (OpenAIServingTranslation, "create_translation"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_synthetic_acceptance_rejects_speech_before_generation(
    serving_class, method_name: str, stream: bool
):
    serving = serving_class.__new__(serving_class)
    serving.synthetic_output = True
    serving.engine_client = Mock()
    request = SimpleNamespace(response_format="json", stream=stream)

    response = await getattr(serving, method_name)(b"audio", request)

    assert isinstance(response, ErrorResponse)
    assert "Speech-to-text" in response.error.message
    serving.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_synthetic_acceptance_rejects_realtime_connection():
    websocket = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                openai_serving_realtime=SimpleNamespace(synthetic_output=True)
            )
        ),
        close=AsyncMock(),
    )

    await realtime_endpoint(websocket)

    websocket.close.assert_awaited_once_with(
        code=1008,
        reason="Realtime transcription is unsupported with synthetic acceptance",
    )
