# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for AuthenticationMiddleware.

These are fast, in-process tests that exercise the ASGI middleware
directly without starting a server.  The primary goal is to verify
that the middleware handles every ASGI scope type correctly —
especially WebSocket scopes, which lack a "method" key.
"""

from collections.abc import Awaitable

import pytest

from vllm.entrypoints.openai.server_utils import AuthenticationMiddleware

API_KEY = "test-secret"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _Recorder:
    """Tiny ASGI app that records whether it was called."""

    def __init__(self):
        self.called = False

    def __call__(self, scope, receive, send) -> Awaitable[None]:
        self.called = True

        async def _noop():
            pass

        return _noop()


def _make_middleware(
    tokens: list[str] | None = None,
) -> tuple[AuthenticationMiddleware, _Recorder]:
    inner = _Recorder()
    mw = AuthenticationMiddleware(inner, tokens or [API_KEY])
    return mw, inner


def _make_asgi_io():
    """Return (receive, send, get_response) for capturing ASGI output."""
    import json

    status_code = None
    body_parts: list[bytes] = []

    async def _receive():
        return {"type": "http.request", "body": b""}

    async def _send(message):
        nonlocal status_code
        mtype = message.get("type", "")
        if mtype == "http.response.start":
            status_code = message["status"]
        elif mtype == "http.response.body":
            body_parts.append(message.get("body", b""))

    def get_response() -> tuple[int | None, dict]:
        body = json.loads(b"".join(body_parts)) if body_parts else {}
        return status_code, body

    return _receive, _send, get_response


def _http_scope(
    path: str = "/v1/models",
    method: str = "GET",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "server": ("localhost", 8000),
        "headers": headers or [],
    }


def _websocket_scope(
    path: str = "/v1/realtime", headers: list[tuple[bytes, bytes]] | None = None
) -> dict:
    """Build a minimal ASGI WebSocket scope — notably, no 'method' key."""
    return {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": path,
        "query_string": b"",
        "root_path": "",
        "server": ("localhost", 8000),
        "headers": headers or [],
    }


def _bearer_header(token: str) -> list[tuple[bytes, bytes]]:
    return [(b"authorization", f"Bearer {token}".encode())]


def _lifespan_scope() -> dict:
    return {"type": "lifespan", "asgi": {"version": "3.0"}}


# ------------------------------------------------------------------
# Tests — WebSocket scopes (the bug)
# ------------------------------------------------------------------


class TestWebSocketAuth:
    """Verify that WebSocket scopes (no 'method' key) don't crash."""

    @pytest.mark.asyncio
    async def test_websocket_with_valid_token_is_forwarded(self):
        mw, inner = _make_middleware()
        scope = _websocket_scope(headers=_bearer_header(API_KEY))
        receive, send, _ = _make_asgi_io()

        await mw(scope, receive, send)

        assert inner.called

    @pytest.mark.asyncio
    async def test_websocket_without_token_is_rejected(self):
        mw, inner = _make_middleware()
        scope = _websocket_scope()
        receive, send, get_response = _make_asgi_io()

        await mw(scope, receive, send)

        assert not inner.called
        status, body = get_response()
        assert status == 401
        assert body == {"error": "Unauthorized"}

    @pytest.mark.asyncio
    async def test_websocket_with_wrong_token_is_rejected(self):
        mw, inner = _make_middleware()
        scope = _websocket_scope(headers=_bearer_header("wrong-key"))
        receive, send, get_response = _make_asgi_io()

        await mw(scope, receive, send)

        assert not inner.called
        status, _ = get_response()
        assert status == 401

    @pytest.mark.asyncio
    async def test_websocket_non_v1_path_skips_auth(self):
        mw, inner = _make_middleware()
        scope = _websocket_scope(path="/health")
        receive, send, _ = _make_asgi_io()

        await mw(scope, receive, send)

        assert inner.called


# ------------------------------------------------------------------
# Tests — HTTP scopes (regression guard)
# ------------------------------------------------------------------


class TestHttpAuth:
    """Existing HTTP auth behaviour must not regress."""

    @pytest.mark.asyncio
    async def test_http_valid_token(self):
        mw, inner = _make_middleware()
        scope = _http_scope(headers=_bearer_header(API_KEY))
        receive, send, _ = _make_asgi_io()

        await mw(scope, receive, send)

        assert inner.called

    @pytest.mark.asyncio
    async def test_http_missing_token(self):
        mw, inner = _make_middleware()
        scope = _http_scope()
        receive, send, get_response = _make_asgi_io()

        await mw(scope, receive, send)

        assert not inner.called
        status, _ = get_response()
        assert status == 401

    @pytest.mark.asyncio
    async def test_http_options_skips_auth(self):
        mw, inner = _make_middleware()
        scope = _http_scope(method="OPTIONS")
        receive, send, _ = _make_asgi_io()

        await mw(scope, receive, send)

        assert inner.called

    @pytest.mark.asyncio
    async def test_http_non_v1_path_skips_auth(self):
        mw, inner = _make_middleware()
        scope = _http_scope(path="/health")
        receive, send, _ = _make_asgi_io()

        await mw(scope, receive, send)

        assert inner.called


# ------------------------------------------------------------------
# Tests — non-HTTP/WebSocket scopes
# ------------------------------------------------------------------


class TestLifespanScope:
    @pytest.mark.asyncio
    async def test_lifespan_scope_is_forwarded(self):
        mw, inner = _make_middleware()
        scope = _lifespan_scope()
        receive, send, _ = _make_asgi_io()

        await mw(scope, receive, send)

        assert inner.called
