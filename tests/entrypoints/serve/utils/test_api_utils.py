# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.entrypoints.generate.base.protocol import StreamOptions
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.utils import api_utils
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    should_include_usage,
)

pytestmark = pytest.mark.skip_global_cleanup


class JsonPayload(BaseModel):
    text: str
    values: dict[int, int]


class NumpyPayload(BaseModel):
    values: Any


@pytest.mark.parametrize("as_model", [False, True])
def test_create_json_response_serializes_numpy(as_model):
    values = np.array([[1.25, -2.5], [0.0, 3.0]], dtype=np.float32)
    result = NumpyPayload(values=values) if as_model else {"values": values}

    response = api_utils.create_json_response(result)

    assert response.status_code == 200
    assert json.loads(response.body) == {"values": values.tolist()}


@pytest.mark.asyncio
async def test_load_aware_call_tracks_json_response_until_completion():
    app = FastAPI()
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    request = Request({"type": "http", "app": app})

    @api_utils.load_aware_call
    async def handler(raw_request: Request):
        return api_utils.create_json_response(JsonPayload(text="ok", values={1: 2}))

    response = await handler(raw_request=request)

    assert app.state.server_load_metrics == 1
    assert response.background is not None
    await response.background()
    assert app.state.server_load_metrics == 0


@pytest.mark.parametrize("container", [None, list, tuple])
def test_create_json_response_serializes_models(container):
    model = JsonPayload(text="测试模型", values={1: 2})
    result = container([model, model]) if container else model
    content = (
        [model.model_dump(), model.model_dump()] if container else model.model_dump()
    )

    response = api_utils.create_json_response(result)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert int(response.headers["content-length"]) == len(response.body)
    assert json.loads(response.body) == json.loads(JSONResponse(content=content).body)


def test_create_json_response_serializes_empty_list():
    response = api_utils.create_json_response([])

    assert response.status_code == 200
    assert response.body == b"[]"


@pytest.mark.parametrize("multiple", [False, True])
def test_create_json_response_preserves_large_integers(multiple):
    model = JsonPayload(text="opaque identifier", values={1: 2**80})

    response = api_utils.create_json_response([model] if multiple else model)

    data = json.loads(response.body)
    item = data[0] if multiple else data
    assert response.status_code == 200
    assert item["values"]["1"] == 2**80


@pytest.mark.parametrize("value", [1, 2**80])
def test_create_json_response_preserves_custom_status(value):
    model = JsonPayload(text="created", values={1: value})

    response = api_utils.create_json_response(model, status_code=201)

    assert response.status_code == 201
    assert json.loads(response.body)["values"]["1"] == value


def test_create_json_response_preserves_error_status():
    error = ErrorResponse(
        error=ErrorInfo(message="invalid prompt", type="BadRequestError", code=400)
    )

    response = api_utils.create_json_response(error)

    assert response.status_code == 400
    assert json.loads(response.body) == error.model_dump()


@pytest.mark.asyncio
async def test_with_cancellation_disconnect_skips_response_validation():
    """A disconnect must cancel the handler without a response_model 500."""
    app = FastAPI()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @app.post("/", response_model=str)
    @api_utils.with_cancellation
    async def handler(body: dict, raw_request: Request):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    messages = []
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert messages[0]["status"] == 200
    assert messages[1]["body"] == b""
    await asyncio.wait_for(cancelled.wait(), timeout=5)


@pytest.mark.asyncio
async def test_with_cancellation_cleans_up_children_when_wrapper_is_cancelled():
    blocker = asyncio.Event()
    handler_started = asyncio.Event()
    receive_started = asyncio.Event()
    child_tasks: list[asyncio.Task] = []

    async def handler(_self, _request):
        task = asyncio.current_task()
        assert task is not None
        child_tasks.append(task)
        handler_started.set()
        await blocker.wait()

    async def receive():
        task = asyncio.current_task()
        assert task is not None
        child_tasks.append(task)
        receive_started.set()
        await blocker.wait()

    wrapped = api_utils.with_cancellation(handler)
    request = Request({"type": "http"}, receive=receive)
    parent_task = asyncio.create_task(wrapped(None, request))
    await asyncio.gather(handler_started.wait(), receive_started.wait())

    try:
        parent_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent_task

        assert len(child_tasks) == 2
        assert all(task.done() for task in child_tasks)
    finally:
        for task in child_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*child_tasks, return_exceptions=True)


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )
