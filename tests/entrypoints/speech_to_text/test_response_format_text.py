# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.speech_to_text.transcription.api_router import (
    router as transcription_router,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponse,
    TranscriptionUsageAudio,
)
from vllm.entrypoints.speech_to_text.translation.api_router import (
    router as translation_router,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationResponse


class _StubSpeechToTextServing:
    def __init__(self, response: TranscriptionResponse | TranslationResponse) -> None:
        self.response = response

    async def create_transcription(self, *args: Any) -> TranscriptionResponse:
        assert isinstance(self.response, TranscriptionResponse)
        return self.response

    async def create_translation(self, *args: Any) -> TranslationResponse:
        assert isinstance(self.response, TranslationResponse)
        return self.response


@pytest.mark.parametrize(
    ("router", "handler_attr", "path", "response"),
    [
        (
            transcription_router,
            "openai_serving_transcription",
            "/v1/audio/transcriptions",
            TranscriptionResponse(
                text="The transcribed text.",
                usage=TranscriptionUsageAudio(seconds=1),
            ),
        ),
        (
            translation_router,
            "openai_serving_translation",
            "/v1/audio/translations",
            TranslationResponse(text="The translated text."),
        ),
    ],
)
@pytest.mark.parametrize("response_format", ["text", "json"])
def test_non_streaming_response_format(
    router,
    handler_attr,
    path,
    response,
    response_format,
):
    app = FastAPI()
    app.include_router(router)
    setattr(app.state, handler_attr, _StubSpeechToTextServing(response))

    client = TestClient(app)
    result = client.post(
        path,
        data={"response_format": response_format},
        files={"file": ("audio.wav", b"audio", "audio/wav")},
    )

    if response_format == "text":
        assert result.headers["content-type"] == "text/plain; charset=utf-8"
        assert result.text == response.text
    else:
        assert result.headers["content-type"] == "application/json"
        assert result.json() == response.model_dump()
