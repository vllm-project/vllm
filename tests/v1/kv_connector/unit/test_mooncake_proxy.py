# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_response_media_type_matches_payload(monkeypatch, endpoint, stream):
    """Clients must select SSE parsing for streaming completion responses."""
    path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    proxy = runpy.run_path(str(path))["_handle_completions"].__globals__
    app = proxy["app"]
    app.state.ready = asyncio.Event()
    app.state.ready.set()
    app.state.prefill_iterator = itertools.cycle([({}, 0)])
    app.state.decode_iterator = itertools.cycle([0])
    app.state.decode_clients = [{}]
    payload = b'{"choices":[{"text":"hello"}]}'
    body = b"data: " + payload + b"\n\ndata: [DONE]\n\n" if stream else payload

    async def decode_response(*args, **kwargs):
        yield body

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    monkeypatch.setitem(proxy, "stream_service_response", decode_response)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post(endpoint, json={"stream": stream})

    expected_type = "text/event-stream" if stream else "application/json"
    assert response.headers["content-type"].split(";")[0] == expected_type
    assert response.content == body
