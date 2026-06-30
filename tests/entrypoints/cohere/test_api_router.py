# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/entrypoints/cohere/api_router.py``.

Covers:

* The optional-import guard: ``attach_router`` is a no-op when the
  ``cohere`` SDK isn't installed.
* The router wiring: response shapes (JSON + SSE), error translation,
  and the ``cohere_serving_chat_v2 is None`` fallback (501 Not
  Implemented).
"""

import json
import sys
from collections.abc import AsyncGenerator
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.cohere.api_router import attach_router
from vllm.entrypoints.cohere.protocol import (
    AssistantMessageResponse,
    CohereChatV2Response,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _Handler:
    """Minimal stand-in for :class:`CohereServingChatV2` used by the
    router. Each test sets ``self.result`` to either:

    * a :class:`CohereChatV2Response` (non-streaming JSON path);
    * an async generator yielding SSE frames (streaming path);
    * an :class:`ErrorResponse` (error envelope path); or
    * an exception (router-level 500 path).
    """

    def __init__(self, result):
        self.result = result

    async def create_chat_v2(self, request, raw_request):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _build_app(handler: _Handler | None) -> FastAPI:
    app = FastAPI()
    attach_router(app)
    app.state.cohere_serving_chat_v2 = handler
    return app


def _minimal_request_body() -> dict:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }


# ----------------------------------------------------------------------
# Optional-import guard
# ----------------------------------------------------------------------


class TestOptionalCohereImport:
    def test_attach_router_noop_when_cohere_missing(self, monkeypatch, caplog):
        # Simulate ``import cohere`` failing by injecting a sentinel that
        # raises ImportError on first attribute access. The simplest
        # approach: stash the real module, delete it from sys.modules,
        # and shadow it with an entry that raises on next import.
        real = sys.modules.pop("cohere", None)

        # Block re-import.
        class _RaisingFinder:
            def find_spec(self, name, path=None, target=None):
                if name == "cohere":
                    raise ImportError("simulated missing cohere SDK")
                return None

        finder = _RaisingFinder()
        sys.meta_path.insert(0, finder)
        try:
            with caplog.at_level("INFO", logger="vllm.entrypoints.cohere.api_router"):
                app = FastAPI()
                attach_router(app)
            # No routes should have been registered.
            paths = [r.path for r in app.routes]
            assert "/cohere/v2/chat" not in paths
            assert any(
                "cohere SDK not installed" in rec.message for rec in caplog.records
            )
        finally:
            sys.meta_path.remove(finder)
            if real is not None:
                sys.modules["cohere"] = real

    def test_attach_router_registers_route_when_cohere_present(self):
        app = _build_app(handler=None)
        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/cohere/v2/chat" in paths


# ----------------------------------------------------------------------
# Endpoint behavior
# ----------------------------------------------------------------------


class TestEndpoint:
    def test_501_when_handler_missing(self):
        app = _build_app(handler=None)
        with TestClient(app) as client:
            r = client.post("/cohere/v2/chat", json=_minimal_request_body())
        assert r.status_code == HTTPStatus.NOT_IMPLEMENTED
        body = r.json()
        assert "does not support" in body["message"]
        assert "id" not in body  # excluded by ``exclude_none=True``

    def test_non_streaming_response_is_json(self):
        msg = AssistantMessageResponse(content=[{"type": "text", "text": "hello"}])
        result = CohereChatV2Response(id="r1", finish_reason="COMPLETE", message=msg)
        app = _build_app(handler=_Handler(result))
        with TestClient(app) as client:
            r = client.post("/cohere/v2/chat", json=_minimal_request_body())
        assert r.status_code == HTTPStatus.OK
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["id"] == "r1"
        assert body["finish_reason"] == "COMPLETE"
        assert body["message"]["content"][0]["text"] == "hello"

    def test_streaming_response_is_sse(self):
        async def _gen() -> AsyncGenerator[str, None]:
            yield 'data: {"type":"message-start"}\n\n'
            yield "data: [DONE]\n\n"

        app = _build_app(handler=_Handler(_gen()))
        with TestClient(app) as client:
            r = client.post(
                "/cohere/v2/chat",
                json={**_minimal_request_body(), "stream": True},
            )
        assert r.status_code == HTTPStatus.OK
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.text
        assert "message-start" in body
        assert body.rstrip().endswith("[DONE]")

    def test_error_response_translated_to_cohere_envelope(self):
        err = ErrorResponse(
            error=ErrorInfo(
                message="bad request",
                type="bad_request",
                code=400,
            )
        )
        app = _build_app(handler=_Handler(err))
        with TestClient(app) as client:
            r = client.post("/cohere/v2/chat", json=_minimal_request_body())
        assert r.status_code == HTTPStatus.BAD_REQUEST
        body = r.json()
        assert body == {"message": "bad request"}

    def test_handler_exception_returns_500_envelope(self):
        app = _build_app(handler=_Handler(RuntimeError("kaboom")))
        with TestClient(app) as client:
            r = client.post("/cohere/v2/chat", json=_minimal_request_body())
        assert r.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        body = r.json()
        assert body == {"message": "kaboom"}

    def test_non_json_content_type_rejected(self):
        """The ``validate_json_request`` dependency raises
        ``RequestValidationError`` (HTTP 422) for non-JSON content
        types, matching the behavior of the other vLLM API routers.
        """
        app = _build_app(handler=None)
        with TestClient(app) as client:
            r = client.post(
                "/cohere/v2/chat",
                content=json.dumps(_minimal_request_body()),
                headers={"content-type": "text/plain"},
            )
        assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_invalid_body_returns_422(self):
        # ``model`` is required; omit it to trip Pydantic validation.
        app = _build_app(handler=None)
        with TestClient(app) as client:
            r = client.post(
                "/cohere/v2/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
