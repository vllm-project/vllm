# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine requests that finish with `error`, `abort` or a server length limit
must fail the transcription instead of returning truncated text as a success.
Options a model rejects must fail before any engine request starts."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.speech_to_text.transcription.api_router import router
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.exceptions import (
    GenerationError,
    RetryableRequestError,
    VLLMUnprocessableEntityError,
)
from vllm.model_executor.models.interfaces import StreamingTranscriptionPostProcessor
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.utils.async_utils import collect_from_async_generator

pytestmark = pytest.mark.skip_global_cleanup

REQUEST_ID = "transcribe-outer-request"
CHUNK_IDS = [f"{REQUEST_ID}-0", f"{REQUEST_ID}-1"]


class _StubTranscriptionModel:
    no_space_languages: set[str] = set()

    @classmethod
    def post_process_output(cls, text: str, model_config) -> str:
        return text

    @classmethod
    def get_num_audio_tokens(cls, *args) -> None:
        return None

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        return language

    @classmethod
    def get_generation_prompt(cls, stt_params):
        if stt_params.use_beam_search:
            raise ValueError("This model does not support beam search")
        return {"prompt": "chunk"}


def _output(text: str, finish_reason: str | None, num_tokens: int = 1) -> RequestOutput:
    return RequestOutput(
        request_id="rid",
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=(1,) * num_tokens,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=finish_reason is not None,
    )


async def _yields(*outputs: RequestOutput) -> AsyncGenerator[RequestOutput, None]:
    for output in outputs:
        yield output


def _server(
    monkeypatch,
    generators: dict[str, Callable[[], AsyncGenerator[RequestOutput, None]]],
) -> tuple[OpenAIServingTranscription, list[str], AsyncMock]:
    started: list[str] = []
    abort = AsyncMock()

    async def generate(_prompt, _params, request_id, **_kwargs):
        started.append(request_id)
        async for output in generators[request_id]():
            yield output

    server = OpenAIServingTranscription.__new__(OpenAIServingTranscription)
    server.engine_client = SimpleNamespace(
        errored=False,
        check_admission=Mock(),
        generate=Mock(side_effect=generate),
        abort=abort,
        is_tracing_enabled=AsyncMock(return_value=False),
    )
    server.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = _StubTranscriptionModel
    server.streaming_post_processor_cls = StreamingTranscriptionPostProcessor
    server.enable_force_include_usage = False
    server.default_sampling_params = {}
    server.asr_config = SpeechToTextConfig(max_audio_clip_s=30)
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_maybe_get_adapters", Mock(return_value=None))
    monkeypatch.setattr(
        server,
        "_preprocess_speech_to_text",
        AsyncMock(
            return_value=(
                [{"prompt": f"chunk-{i}"} for i in range(len(generators))],
                1.0,
                [30.0 * i for i in range(len(generators))],
            )
        ),
    )
    monkeypatch.setattr(server, "_log_inputs", Mock())
    return server, started, abort


def _request(
    stream: bool,
    use_beam_search: bool = False,
    max_completion_tokens: int | None = None,
) -> TranscriptionRequest:
    return TranscriptionRequest.model_construct(
        file=Mock(),
        model="audio",
        language="en",
        stream=stream,
        use_beam_search=use_beam_search,
        max_completion_tokens=max_completion_tokens,
        response_format="json",
    )


def _raw_request() -> SimpleNamespace:
    return SimpleNamespace(
        headers={"X-Request-Id": "outer-request"}, state=SimpleNamespace()
    )


def _post(server: OpenAIServingTranscription, **form: str):
    app = FastAPI()
    app.include_router(router)
    init_exception_handler(app)
    app.state.openai_serving_transcription = server
    app.state.args = SimpleNamespace(log_error_stack=False)

    with TestClient(app) as client:
        return client.post(
            "/v1/audio/transcriptions",
            headers={"X-Request-Id": "outer-request"},
            data={"model": "audio", "language": "en", **form},
            files={"file": ("audio.wav", b"audio", "audio/wav")},
        )


@pytest.mark.parametrize(
    ("finish_reason", "client_limit", "status", "body"),
    [
        ("stop", None, HTTPStatus.OK, "hello"),
        ("stop", 5, HTTPStatus.OK, "hello"),
        # The client's own cap stopped the chunk: truncation was requested.
        ("length", 2, HTTPStatus.OK, "hello"),
        # A server cap stopped the chunk before any client cap.
        ("length", None, HTTPStatus.UNPROCESSABLE_ENTITY, "UnprocessableEntityError"),
        ("length", 5, HTTPStatus.UNPROCESSABLE_ENTITY, "UnprocessableEntityError"),
        ("error", None, HTTPStatus.INTERNAL_SERVER_ERROR, "InternalServerError"),
        ("abort", None, HTTPStatus.SERVICE_UNAVAILABLE, "ServiceUnavailableError"),
    ],
)
def test_http_non_streaming_status_follows_finish_reason(
    finish_reason, client_limit, status, body, monkeypatch
):
    server, _, _ = _server(
        monkeypatch,
        {REQUEST_ID: lambda: _yields(_output("hello", finish_reason, num_tokens=2))},
    )
    form = {} if client_limit is None else {"max_completion_tokens": str(client_limit)}

    response = _post(server, **form)

    assert response.status_code == status
    if status == HTTPStatus.OK:
        assert response.json()["text"] == body
    else:
        assert response.json()["error"]["type"] == body
        assert response.json()["error"]["code"] == status


@pytest.mark.asyncio
@pytest.mark.parametrize("use_beam_search", [False, True])
@pytest.mark.parametrize(
    ("finish_reason", "exc_type"),
    [
        ("error", GenerationError),
        ("abort", RetryableRequestError),
        ("length", VLLMUnprocessableEntityError),
    ],
)
async def test_non_streaming_failed_chunk_stops_live_sibling_chunk(
    finish_reason, exc_type, use_beam_search, monkeypatch
):
    sibling_started = asyncio.Event()
    sibling_stopped = asyncio.Event()

    async def failing_chunk() -> AsyncGenerator[RequestOutput, None]:
        await sibling_started.wait()
        yield _output("partial", finish_reason)

    async def live_sibling_chunk() -> AsyncGenerator[RequestOutput, None]:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
            yield _output("", None)
        except (asyncio.CancelledError, GeneratorExit):
            sibling_stopped.set()
            raise

    # Like `BeamSearchOnlineMixin.beam_search`, run each chunk as a sub-request
    # with its own id in a task awaited through `asyncio.gather`; aborting the
    # chunk id does not reach it, so only closing the chunk generator does.
    engine_ids = [f"{chunk_id}-x-beam-0" for chunk_id in CHUNK_IDS]

    async def beam_search(*, prompt, params, request_id, **kwargs):
        sub_request = asyncio.create_task(
            collect_from_async_generator(
                server.engine_client.generate(prompt, params, f"{request_id}-x-beam-0")
            )
        )
        (outputs,) = await asyncio.gather(sub_request)
        for output in outputs:
            yield output

    if not use_beam_search:
        engine_ids = CHUNK_IDS
    server, started, abort = _server(
        monkeypatch,
        {engine_ids[0]: failing_chunk, engine_ids[1]: live_sibling_chunk},
    )
    if use_beam_search:
        monkeypatch.setattr(server, "beam_search", beam_search)

    # Holding the traceback keeps the handler frame alive, so the sibling
    # must stop without waiting for the merged iterator to be collected.
    with pytest.raises(exc_type) as exc_info:
        await asyncio.wait_for(
            server._create_speech_to_text(
                audio_data=b"audio",
                request=_request(stream=False, use_beam_search=use_beam_search),
                raw_request=_raw_request(),
                response_class=Mock(),
                stream_generator_method=Mock(),
            ),
            timeout=10,
        )

    await asyncio.wait_for(sibling_stopped.wait(), timeout=10)
    assert exc_info.type is exc_type
    assert started == engine_ids
    abort.assert_awaited_once_with(CHUNK_IDS)


def _events(lines: list[str]) -> list[Any]:
    events: list[Any] = []
    for line in lines:
        payload = line.removeprefix("data: ").strip()
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


async def _stream(
    server: OpenAIServingTranscription, max_completion_tokens: int | None = None
) -> list[Any]:
    generator = await server._create_speech_to_text(
        audio_data=b"audio",
        request=_request(stream=True, max_completion_tokens=max_completion_tokens),
        raw_request=_raw_request(),
        response_class=Mock(),
        stream_generator_method=server.transcription_stream_generator,
    )
    assert isinstance(generator, AsyncGenerator)
    return _events([line async for line in generator])


@pytest.mark.asyncio
async def test_streaming_error_emits_error_event_and_stops(monkeypatch):
    server, started, _ = _server(
        monkeypatch,
        {
            CHUNK_IDS[0]: lambda: _yields(_output("hel", None), _output("lo", "error")),
            CHUNK_IDS[1]: lambda: _yields(_output("world", "stop")),
        },
    )

    events = await _stream(server)

    assert events[0]["choices"][0]["delta"]["content"] == "hel"
    assert events[1]["error"]["code"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert events[1]["error"]["type"] == "InternalServerError"
    assert events[2:] == ["[DONE]"]
    assert started == CHUNK_IDS[:1]


@pytest.mark.asyncio
async def test_streaming_aborted_chunk_skips_remaining_chunks(monkeypatch):
    server, started, abort = _server(
        monkeypatch,
        {
            CHUNK_IDS[0]: lambda: _yields(_output("hel", None), _output("lo", "abort")),
            CHUNK_IDS[1]: lambda: _yields(_output("world", "stop")),
        },
    )

    events = await _stream(server)

    assert events[0]["choices"][0]["delta"]["content"] == "hel"
    assert events[1]["choices"][0]["delta"]["content"] == "lo"
    assert events[1]["choices"][0]["finish_reason"] == "abort"
    assert events[2:] == ["[DONE]"]
    assert started == CHUNK_IDS[:1]
    abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_server_length_limit_emits_error_event_and_stops(
    monkeypatch,
):
    server, started, _ = _server(
        monkeypatch,
        {
            CHUNK_IDS[0]: lambda: _yields(
                _output("hel", None), _output("lo", "length")
            ),
            CHUNK_IDS[1]: lambda: _yields(_output("world", "stop")),
        },
    )

    events = await _stream(server, max_completion_tokens=5)

    assert events[0]["choices"][0]["delta"]["content"] == "hel"
    assert events[1]["error"]["code"] == HTTPStatus.UNPROCESSABLE_ENTITY
    assert events[1]["error"]["type"] == "UnprocessableEntityError"
    assert events[2:] == ["[DONE]"]
    assert started == CHUNK_IDS[:1]


@pytest.mark.asyncio
async def test_streaming_client_length_limit_finishes_normally(monkeypatch):
    """The client cap counts tokens across all deltas of a chunk."""
    server, started, _ = _server(
        monkeypatch,
        {
            CHUNK_IDS[0]: lambda: _yields(
                _output("hel", None), _output("lo", "length")
            ),
            CHUNK_IDS[1]: lambda: _yields(_output("world", "stop")),
        },
    )

    events = await _stream(server, max_completion_tokens=2)

    assert [e["choices"][0]["delta"]["content"] for e in events[:3]] == [
        "hel",
        "lo",
        " world",
    ]
    assert events[1]["choices"][0]["finish_reason"] == "length"
    assert events[3:] == ["[DONE]"]
    assert started == CHUNK_IDS


def test_http_model_rejected_option_returns_400_before_engine(monkeypatch):
    server, started, _ = _server(
        monkeypatch,
        {REQUEST_ID: lambda: _yields(_output("hello", "stop"))},
    )
    monkeypatch.delattr(server, "_preprocess_speech_to_text")
    server._decode_and_chunk_speech_async = AsyncMock(
        return_value=([np.zeros(160, dtype=np.float32)], 0.01)
    )

    response = _post(server, use_beam_search="true")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "beam search" in response.json()["error"]["message"]
    assert started == []
