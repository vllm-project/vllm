# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def proxy():
    path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    proxy = runpy.run_path(str(path))["_handle_completions"].__globals__
    app = proxy["app"]
    app.state.ready = asyncio.Event()
    app.state.ready.set()
    app.state.prefill_iterator = itertools.cycle(
        [
            (
                {
                    "bootstrap_addr": "http://prefill:8998",
                    "dp_engine_id": {0: "engine"},
                },
                0,
            )
        ]
    )
    app.state.decode_iterator = itertools.cycle([0])
    return proxy


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 500])
async def test_decode_response_is_forwarded(
    monkeypatch, proxy, endpoint, stream, status_code
):
    """Forward decode status, content type, and bytes through the real route."""
    app = proxy["app"]
    if status_code == 500:
        body = b'{"error":{"message":"controlled failure"}}'
        content_type = "application/json"
    elif stream:
        body = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        content_type = "text/event-stream"
    else:
        body = b'{"choices":[]}'
        content_type = "application/json"
    closed = []

    class DecodeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body[:5]
            yield body[5:]

        async def aclose(self):
            closed.append(True)

    async def decode_backend(request):
        assert request.url.path == endpoint
        assert request.method == "POST"
        assert json.loads(request.content)["stream"] is stream
        return httpx.Response(
            status_code,
            headers={"content-type": content_type},
            stream=DecodeBody(),
        )

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(decode_backend),
        base_url="http://decode",
    ) as decode_client:
        app.state.decode_clients = [{"client": decode_client}]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://proxy",
        ) as client:
            response = await client.post(endpoint, json={"stream": stream})

    assert response.status_code == status_code
    assert response.headers["content-type"] == content_type
    assert response.content == body
    assert closed == [True]


@pytest.mark.asyncio
async def test_upstream_response_closes_with_downstream_stream(monkeypatch, proxy):
    app = proxy["app"]
    closed = []

    class DecodeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"chunk"

        async def aclose(self):
            closed.append(True)

    async def decode_backend(request):
        return httpx.Response(200, stream=DecodeBody())

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(decode_backend),
        base_url="http://decode",
    ) as decode_client:
        app.state.decode_clients = [{"client": decode_client}]
        request_body = json.dumps({"stream": True}).encode()

        async def receive():
            return {"type": "http.request", "body": request_body}

        request = proxy["Request"](
            {
                "type": "http",
                "app": app,
                "method": "POST",
                "path": "/v1/completions",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
        )
        response = await proxy["_handle_completions"]("/v1/completions", request)
        assert await anext(response.body_iterator) == b"chunk"
        await response.body_iterator.aclose()

    assert closed == [True]
