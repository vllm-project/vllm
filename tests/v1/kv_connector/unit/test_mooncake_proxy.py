# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from starlette.requests import ClientDisconnect

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
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_upstream_response_closes_when_downstream_headers_fail(
    monkeypatch, proxy, endpoint, spec_version
):
    """Close the opened decode response even when body iteration never starts."""
    app = proxy["app"]
    body_started = []
    closed = []
    headers_attempted = []

    class DecodeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            body_started.append(True)
            yield b"chunk"

        async def aclose(self):
            closed.append(True)

    upstream = httpx.Response(200, stream=DecodeBody())
    request_received = False

    async def decode_backend(request):
        assert request.url.path == endpoint
        return upstream

    async def receive():
        nonlocal request_received
        if request_received:
            await asyncio.Event().wait()
        request_received = True
        return {"type": "http.request", "body": b'{"stream": true}'}

    async def send(message):
        assert message["type"] == "http.response.start"
        headers_attempted.append(True)
        raise OSError("downstream header send failed")

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(decode_backend),
        base_url="http://decode",
    ) as decode_client:
        app.state.decode_clients = [{"client": decode_client}]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": endpoint,
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        }
        try:
            expected_error = OSError if spec_version == "2.3" else ClientDisconnect
            with pytest.raises(expected_error):
                await app(scope, receive, send)
            assert headers_attempted == [True]
            assert body_started == []
            assert upstream.is_closed
            assert closed == [True]
        finally:
            # Do not let test teardown hide a missing response close.
            await upstream.aclose()


@pytest.mark.asyncio
async def test_upstream_cleanup_finishes_on_asgi23_disconnect(monkeypatch, proxy):
    """Disconnect cancellation must not interrupt asynchronous upstream cleanup."""
    app = proxy["app"]
    body_sent = asyncio.Event()
    cleanup_finished = []
    request_received = False

    class DecodeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"chunk"
            await asyncio.Event().wait()

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            cleanup_finished.append(True)

    upstream = httpx.Response(200, stream=DecodeBody())

    async def decode_backend(request):
        return upstream

    async def receive():
        nonlocal request_received
        if not request_received:
            request_received = True
            return {"type": "http.request", "body": b'{"stream": true}'}
        await body_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            body_sent.set()

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(decode_backend),
        base_url="http://decode",
    ) as decode_client:
        app.state.decode_clients = [{"client": decode_client}]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        }
        try:
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
            assert body_sent.is_set()
            assert upstream.is_closed
            assert cleanup_finished == [True]
        finally:
            await upstream.aclose()


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
