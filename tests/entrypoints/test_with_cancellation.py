# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from typing import Annotated, Any

import pytest
from fastapi import Body, FastAPI, Request

from vllm.entrypoints.utils import with_cancellation


class _MockAppState:
    enable_server_load_tracking = False


class _MockApp:
    state = _MockAppState()


class _DisconnectingRequest:
    app = _MockApp()

    async def receive(self) -> dict[str, str]:
        return {"type": "http.disconnect"}


@pytest.mark.asyncio
async def test_with_cancellation_raises_on_disconnect() -> None:
    @with_cancellation
    async def handler(raw_request: Request):
        await asyncio.sleep(60)
        return {"ok": True}

    with pytest.raises(asyncio.CancelledError, match="Client disconnected"):
        await handler(raw_request=_DisconnectingRequest())

    # Let the event loop process cancellation of the pending handler task.
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_with_cancellation_disconnect_does_not_emit_200_null() -> None:
    app = FastAPI()

    @app.post("/repro")
    @with_cancellation
    async def repro_route(
        raw_request: Request,
        payload: Annotated[dict[str, str], Body()],
    ):
        await asyncio.sleep(60)
        return {"ok": True, "payload": payload}

    body = json.dumps({"message": "body was fully received"}).encode()
    request_messages: list[dict[str, Any]] = [
        {"type": "http.request", "body": body, "more_body": False},
        {"type": "http.disconnect"},
    ]
    response_messages: list[dict[str, Any]] = []

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/repro",
        "raw_path": b"/repro",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }

    async def receive() -> dict[str, Any]:
        if request_messages:
            return request_messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        response_messages.append(message)

    with pytest.raises(asyncio.CancelledError, match="Client disconnected"):
        await app(scope, receive, send)

    await asyncio.sleep(0)

    status = None
    body_parts = []
    for message in response_messages:
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body_parts.append(message.get("body", b""))

    assert (status, b"".join(body_parts)) != (200, b"null")
