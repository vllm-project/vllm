# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.speech_to_text.transcription.api_router import (
    router as transcription_router,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionUsageAudio,
)
from vllm.entrypoints.speech_to_text.translation.api_router import (
    router as translation_router,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationResponse,
    TranslationResponseVerbose,
)

pytestmark = pytest.mark.cpu_test

SAMPLE_TEXT = '  Né più mai toccherò — "hello" 你好\n'


class _StubSpeechToTextServing:
    def __init__(self, result: Any) -> None:
        self.result = result

    async def create_transcription(self, audio_data, request, raw_request):
        return self.result

    async def create_translation(self, audio_data, request, raw_request):
        return self.result


ENDPOINTS = [
    pytest.param(
        transcription_router,
        "openai_serving_transcription",
        "/v1/audio/transcriptions",
        lambda text: TranscriptionResponse(
            text=text,
            usage=TranscriptionUsageAudio(seconds=3),
        ),
        lambda text: TranscriptionResponseVerbose(
            duration=3.0,
            language="en",
            text=text,
            segments=[],
        ),
        id="transcriptions",
    ),
    pytest.param(
        translation_router,
        "openai_serving_translation",
        "/v1/audio/translations",
        lambda text: TranslationResponse(text=text),
        lambda text: TranslationResponseVerbose(
            duration=3.0,
            language="it",
            text=text,
            segments=[],
        ),
        id="translations",
    ),
]


def _make_client(
    router,
    state_attr: str,
    result: Any,
    *,
    enable_server_load_tracking: bool = False,
) -> tuple[FastAPI, TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.state.enable_server_load_tracking = enable_server_load_tracking
    if enable_server_load_tracking:
        app.state.server_load_metrics = 0
    setattr(app.state, state_attr, _StubSpeechToTextServing(result))
    return app, TestClient(app)


def _post_audio(client: TestClient, path: str, **form_data: str):
    return client.post(
        path,
        data={"model": "whisper-test", **form_data},
        files={"file": ("sample.wav", b"\x00" * 16, "audio/wav")},
    )


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
@pytest.mark.parametrize("text", [SAMPLE_TEXT, ""])
def test_response_format_text_returns_plain_text(
    router, state_attr, path, make_response, make_verbose, text
):
    _, client = _make_client(router, state_attr, make_response(text))

    response = _post_audio(client, path, response_format="text")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.content == text.encode("utf-8")


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
@pytest.mark.parametrize("form_data", [{}, {"response_format": "json"}])
def test_response_format_json_and_default_unchanged(
    router, state_attr, path, make_response, make_verbose, form_data
):
    expected = make_response(SAMPLE_TEXT)
    _, client = _make_client(router, state_attr, expected)

    response = _post_audio(client, path, **form_data)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == expected.model_dump()


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
def test_response_format_verbose_json_unchanged(
    router, state_attr, path, make_response, make_verbose
):
    expected = make_verbose(SAMPLE_TEXT)
    _, client = _make_client(router, state_attr, expected)

    response = _post_audio(client, path, response_format="verbose_json")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == expected.model_dump()


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
def test_response_format_text_error_remains_json(
    router, state_attr, path, make_response, make_verbose
):
    error = ErrorResponse(
        error=ErrorInfo(
            message="Unsupported audio",
            type="BadRequestError",
            code=400,
        )
    )
    _, client = _make_client(router, state_attr, error)

    response = _post_audio(client, path, response_format="text")

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == error.model_dump()


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
def test_response_format_text_streaming_remains_sse(
    router, state_attr, path, make_response, make_verbose
):
    async def _stream() -> AsyncGenerator[str, None]:
        yield 'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        yield "data: [DONE]\n\n"

    _, client = _make_client(router, state_attr, _stream())

    response = _post_audio(client, path, response_format="text", stream="true")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'data: {"choices": [{"delta": {"content": "hello"}}]}\n\ndata: [DONE]\n\n'
    )


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
def test_response_format_text_decrements_server_load_metrics(
    router, state_attr, path, make_response, make_verbose
):
    app, client = _make_client(
        router,
        state_attr,
        make_response("hello"),
        enable_server_load_tracking=True,
    )

    response = _post_audio(client, path, response_format="text")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert app.state.server_load_metrics == 0


@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
def test_openapi_documents_text_plain_response(
    router, state_attr, path, make_response, make_verbose
):
    app, _ = _make_client(router, state_attr, make_response("hello"))
    content = app.openapi()["paths"][path]["post"]["responses"]["200"]["content"]

    assert "text/plain" in content
    assert content["text/plain"]["schema"] == {"type": "string"}
    assert "application/json" in content
    assert "text/event-stream" in content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router", "state_attr", "path", "make_response", "make_verbose"),
    ENDPOINTS,
)
async def test_openai_sdk_receives_plain_text(
    router, state_attr, path, make_response, make_verbose
):
    app, _ = _make_client(router, state_attr, make_response(SAMPLE_TEXT))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http_client:
        client = AsyncOpenAI(
            api_key="test",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        resource = (
            client.audio.transcriptions
            if path.endswith("/transcriptions")
            else client.audio.translations
        )
        raw_response = await resource.with_raw_response.create(
            model="whisper-test",
            file=("sample.wav", b"\x00" * 16, "audio/wav"),
            response_format="text",
        )

    assert raw_response.headers["content-type"] == "text/plain; charset=utf-8"
    assert raw_response.parse() == SAMPLE_TEXT
