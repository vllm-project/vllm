# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for middleware that's off by default and can be toggled through
server arguments, mainly --api-key and --enable-request-id-headers.
"""

import json
from argparse import Namespace
from http import HTTPStatus
from unittest.mock import AsyncMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.serve.middleware.omit_unset_chat_fields import (
    OmitUnsetChatFieldsMiddleware,
)
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware

# Use a small embeddings model for faster startup and smaller memory footprint.
# Since we are not testing any chat functionality,
# using a chat capable model is overkill.
MODEL_NAME = "intfloat/multilingual-e5-small"


@pytest.fixture(scope="module")
def server(request: pytest.FixtureRequest):
    passed_params = []
    if hasattr(request, "param"):
        passed_params = request.param
    if isinstance(passed_params, str):
        passed_params = [passed_params]

    args = [
        "--runner",
        "pooling",
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "2",
        *passed_params,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_no_api_token(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("v1/models"))
    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_no_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))
    assert "X-Request-Id" not in response.headers


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_missing_api_token(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("v1/models"))
    assert response.status_code == HTTPStatus.UNAUTHORIZED


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_passed_api_token(server: RemoteOpenAIServer):
    response = requests.get(
        server.url_for("v1/models"), headers={"Authorization": "Bearer test"}
    )
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_not_v1_or_v2_path_skips_auth(server: RemoteOpenAIServer):
    # Authorization check is skipped for paths that
    # don't start with /v1 or /v2 (e.g. /health, /metrics).
    response = requests.get(server.url_for("health"))
    assert response.status_code == HTTPStatus.OK


# ---------------------------------------------------------------------------
# /v2 path authentication tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_v2_endpoint_rejects_missing_api_token(server: RemoteOpenAIServer):
    # /v2/embed should require authentication when --api-key is set.
    body = {
        "model": MODEL_NAME,
        "texts": ["hello"],
        "embedding_types": ["float"],
    }
    response = requests.post(server.url_for("/v2/embed"), json=body)
    assert response.status_code == HTTPStatus.UNAUTHORIZED


@pytest.mark.parametrize(
    "server",
    [["--api-key", "test"]],
    indirect=True,
)
@pytest.mark.asyncio
async def test_v2_endpoint_accepts_valid_api_token(server: RemoteOpenAIServer):
    # /v2/embed should accept requests with a valid API key.
    body = {
        "model": MODEL_NAME,
        "texts": ["hello"],
        "embedding_types": ["float"],
    }
    response = requests.post(
        server.url_for("/v2/embed"),
        json=body,
        headers={"Authorization": "Bearer test"},
    )
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "server",
    ["--enable-request-id-headers"],
    indirect=True,
)
@pytest.mark.asyncio
async def test_enable_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))
    assert "X-Request-Id" in response.headers
    assert len(response.headers.get("X-Request-Id", "")) == 32


@pytest.mark.parametrize(
    "server",
    ["--enable-request-id-headers"],
    indirect=True,
)
@pytest.mark.asyncio
async def test_custom_request_id_header(server: RemoteOpenAIServer):
    response = requests.get(
        server.url_for("health"), headers={"X-Request-Id": "Custom"}
    )
    assert "X-Request-Id" in response.headers
    assert response.headers.get("X-Request-Id") == "Custom"


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("enabled", "path"),
    [
        (False, "/v1/chat/completions"),
        (True, "/v1/chat/completions"),
        (True, "/v1/chat/completions/batch"),
    ],
)
def test_omit_unset_chat_fields_is_optional(enabled, path, monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    populated_fields = {
        "prompt_logprobs": [],
        "prompt_token_ids": [1, 2],
        "prompt_text": "",
        "kv_transfer_params": {},
        "ec_transfer_params": {},
        "metrics": {"time_to_first_token_ms": 1.0},
    }
    openai_fields = {
        "service_tier": None,
        "system_fingerprint": None,
        "choices": [{"message": {"content": "hello"}, "logprobs": None}],
    }
    payload = openai_fields | dict.fromkeys(populated_fields)
    app = FastAPI()

    @app.post(path)
    async def completion():
        return payload

    args = Namespace(
        allowed_origins=[],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[
            "vllm.entrypoints.serve.middleware.omit_unset_chat_fields."
            "OmitUnsetChatFieldsMiddleware"
        ]
        if enabled
        else [],
    )
    init_entrypoints_middleware(args, app, supported_tasks=())
    with TestClient(app) as client:
        response = client.post(path)
        assert response.status_code == HTTPStatus.OK
        assert response.json() == (openai_fields if enabled else payload)
        payload.update(populated_fields)
        assert client.post(path).json() == payload


@pytest.mark.skip_global_cleanup
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("root_path", "path"),
    [
        ("/proxy", "/proxy/v1/chat/completions"),
        ("/v", "/v1/chat/completions"),
        ("/", "/v1/chat/completions"),
    ],
)
async def test_omit_unset_chat_fields_handles_multipart_json(root_path, path):
    body = '{"prompt_text": null, "id": "中文"}'.encode()
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode()),
        (b"set-cookie", b"first=1"),
        (b"set-cookie", b"second=2"),
    ]
    original_headers = headers.copy()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body[:-3], "more_body": True})
        await send({"type": "http.response.body", "body": body[-3:]})

    send = AsyncMock()
    await OmitUnsetChatFieldsMiddleware(app)(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "root_path": root_path,
        },
        AsyncMock(),
        send,
    )
    messages = [call.args[0] for call in send.await_args_list]
    result = b"".join(message.get("body", b"") for message in messages)
    assert json.loads(result) == {"id": "中文"}
    result_headers = messages[0]["headers"]
    assert dict(result_headers)[b"content-length"] == str(len(result)).encode()
    assert [(k, v) for k, v in result_headers if k != b"content-length"] == [
        (k, v) for k, v in original_headers if k != b"content-length"
    ]


@pytest.mark.skip_global_cleanup
@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["sse", "error", "other", "compressed", "websocket"])
async def test_omit_unset_chat_fields_bypasses_other_responses(case):
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
    headers = [(b"content-type", b"application/json")]
    if case == "sse":
        headers = [(b"content-type", b"text/event-stream")]
    elif case == "other":
        scope["path"] = "/v1/completions"
    elif case == "compressed":
        headers.append((b"content-encoding", b"gzip"))
    messages = [
        {
            "type": "http.response.start",
            "status": 400 if case == "error" else 200,
            "headers": headers,
        },
        {"type": "http.response.body", "body": b"first", "more_body": True},
        {"type": "http.response.body", "body": b"second"},
    ]
    if case == "websocket":
        scope["type"] = "websocket"
        messages = [{"type": "websocket.accept"}, {"type": "websocket.close"}]
    downstream_send = AsyncMock()

    async def app(scope, receive, send):
        for index, message in enumerate(messages):
            await send(message)
            assert downstream_send.await_count == index + 1

    await OmitUnsetChatFieldsMiddleware(app)(scope, AsyncMock(), downstream_send)
    assert [call.args[0] for call in downstream_send.await_args_list] == messages
