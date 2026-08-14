# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/entrypoints/cohere/api_router.py``.

Covers:

* The optional-import guard: ``attach_router`` is a no-op when the
  ``cohere`` SDK isn't installed.
* The env-var opt-in gate: ``attach_router`` is a no-op unless
  ``VLLM_ENABLE_COHERE_API=1`` is set.
* The router wiring: response shapes (JSON + SSE), error translation,
  and the ``cohere_serving_chat_v2 is None`` fallback (501 Not
  Implemented).
"""

import json
from argparse import Namespace
from collections.abc import AsyncGenerator
from http import HTTPStatus

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from vllm.entrypoints.cohere import api_router as api_router_mod
from vllm.entrypoints.cohere.api_router import attach_router
from vllm.entrypoints.cohere.protocol import (
    AssistantMessageResponse,
    CohereChatV2Response,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.exception_handler.handler.http import http_exception_handler
from vllm.entrypoints.serve.exception_handler.handler.validation import (
    validation_exception_handler,
)


@pytest.fixture(autouse=True)
def _enable_cohere_api(monkeypatch):
    """Auto-enable the Cohere API gate for every test in this module.

    The endpoint is opt-in in production (``VLLM_ENABLE_COHERE_API=1``);
    every test in this file exercises the enabled path *except* the
    dedicated gate test in :class:`TestEnvVarGate`, which unsets the
    flag inside the test body.
    """
    monkeypatch.setenv("VLLM_ENABLE_COHERE_API", "1")


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


def _build_app_with_vllm_handlers(handler: _Handler | None) -> FastAPI:
    """Build a FastAPI app that mirrors the real vLLM setup by installing
    ``validation_exception_handler`` and ``http_exception_handler``. The
    :class:`CohereErrorEnvelopeMiddleware` registered by ``attach_router``
    is expected to translate any resulting vLLM ``ErrorResponse`` body
    into the ``CohereError`` wire shape.
    """
    app = FastAPI()
    attach_router(app)
    app.state.cohere_serving_chat_v2 = handler
    # ``validation_exception_handler`` reads ``req.app.state.args``; the
    # real cli builds this via argparse.
    app.state.args = Namespace(log_error_stack=False)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.exception_handler(HTTPException)(http_exception_handler)
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
    """``attach_router`` probes for the SDK once at module load (because
    the route handler uses types imported from ``cohere``) and stashes
     the result in ``_SDK_AVAILABLE``. Tests simulate the "SDK missing"
    state by flipping that flag for the duration of the test.

    ``attach_router`` checks the env-var gate before the SDK probe, so
    the SDK-missing branch is only reachable when the operator opts in
    via ``VLLM_ENABLE_COHERE_API=1``. The flag-off-and-SDK-missing case
    below exists to pin down that ordering — the flag-off short-circuits
    """

    def test_flag_off_and_sdk_missing_stays_silent_about_sdk(self, monkeypatch, caplog):
        """Flag off doesn't do SDK-missing check.

        When the operator hasn't opted in, ``attach_router`` must not
        warn about the ``cohere`` SDK being missing: they never asked
        for the endpoint, so surfacing the SDK gap is misleading noise.
        Only the flag-off DEBUG message should fire.
        """
        monkeypatch.delenv("VLLM_ENABLE_COHERE_API", raising=False)
        monkeypatch.setattr(api_router_mod, "_SDK_AVAILABLE", False)

        with caplog.at_level("DEBUG", logger="vllm.entrypoints.cohere.api_router"):
            app = FastAPI()
            attach_router(app)

        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/cohere/v2/chat" not in paths
        # The flag-off short-circuit ran; the SDK check never did.
        assert not any(
            "SDK is not installed" in rec.message for rec in caplog.records
        ), "SDK-missing log leaked despite the flag being off"

    def test_flag_on_but_sdk_missing_logs_warning(self, monkeypatch, caplog):
        """Misconfiguration path: the operator explicitly opted into the
        endpoint via ``VLLM_ENABLE_COHERE_API=1`` (already set by the
        autouse fixture) but forgot to install ``cohere``.
        """
        monkeypatch.setattr(api_router_mod, "_SDK_AVAILABLE", False)

        with caplog.at_level("DEBUG", logger="vllm.entrypoints.cohere.api_router"):
            app = FastAPI()
            attach_router(app)

        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/cohere/v2/chat" not in paths
        warn_sdk_records = [
            rec
            for rec in caplog.records
            if "VLLM_ENABLE_COHERE_API=1" in rec.message
            and "SDK is not installed" in rec.message
        ]
        assert warn_sdk_records, (
            "expected a WARNING that pairs the opt-in flag with the "
            "missing SDK so operators notice the misconfiguration"
        )
        assert all(rec.levelname == "WARNING" for rec in warn_sdk_records)

    def test_attach_router_registers_route_when_cohere_present(self):
        app = _build_app(handler=None)
        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/cohere/v2/chat" in paths


# ----------------------------------------------------------------------
# VLLM_ENABLE_COHERE_API gate
# ----------------------------------------------------------------------


class TestEnvVarGate:
    """The Cohere v2 endpoint is opt-in via ``VLLM_ENABLE_COHERE_API``.

    Even with the SDK installed, :func:`attach_router` must skip route
    registration and middleware installation unless the env flag is
    set. The autouse fixture on this module enables the flag by
    default, so each test here explicitly disables it.
    """

    def test_attach_router_noop_when_flag_unset(self, monkeypatch, caplog):
        monkeypatch.delenv("VLLM_ENABLE_COHERE_API", raising=False)

        # The flag-off skip logs at DEBUG on purpose: this is the default
        # state for every non-Cohere vLLM deployment, so an INFO log on
        # every server startup would be pointless noise. The test raises
        # caplog's level accordingly.
        with caplog.at_level("DEBUG", logger="vllm.entrypoints.cohere.api_router"):
            app = FastAPI()
            attach_router(app)

        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/cohere/v2/chat" not in paths
        debug_flag_records = [
            rec
            for rec in caplog.records
            if "VLLM_ENABLE_COHERE_API is not set" in rec.message
        ]
        assert debug_flag_records, (
            "expected a DEBUG message that the cohere flag is off"
        )
        assert all(rec.levelname == "DEBUG" for rec in debug_flag_records)

    def test_attach_router_noop_when_flag_zero(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENABLE_COHERE_API", "0")

        app = FastAPI()
        attach_router(app)

        paths = [getattr(r, "path", None) for r in app.routes]
        assert "/cohere/v2/chat" not in paths


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


# ----------------------------------------------------------------------
# CohereErrorEnvelopeMiddleware
# ----------------------------------------------------------------------


class TestCohereErrorEnvelope:
    """When the app installs vLLM's global exception handlers, validation
    and HTTP errors escape as ``ErrorResponse`` bodies before the route
    handler runs. The middleware installed by ``attach_router`` must
    normalise those bodies to the ``CohereError`` shape declared on the
    endpoint's OpenAPI ``responses`` map so schema-conformance tests
    (``test_openai_schema.py``) don't see a mismatch on ``/cohere/*``
    responses.
    """

    def test_validation_error_body_is_cohere_shaped(self):
        # ``model=""`` and ``messages=[]`` trip our custom field
        # validators, which raise pydantic ValueErrors and are routed
        # through ``validation_exception_handler`` in the real vLLM
        # server (producing the ``{"error": {...}}`` shape).
        app = _build_app_with_vllm_handlers(handler=None)
        with TestClient(app) as client:
            r = client.post("/cohere/v2/chat", json={"messages": [], "model": ""})
        assert r.status_code == HTTPStatus.BAD_REQUEST
        body = r.json()
        # ``CohereError`` has ``message`` at the top level, not nested
        # under an ``error`` envelope.
        assert "error" not in body
        assert "message" in body
        assert isinstance(body["message"], str) and body["message"]

    def test_http_error_body_is_cohere_shaped(self):
        # A raised ``HTTPException`` from anywhere in the request cycle
        # is routed through ``http_exception_handler`` (producing the
        # ``ErrorResponse`` shape) and must be translated.
        app = _build_app_with_vllm_handlers(handler=None)

        @app.get("/cohere/v2/boom")
        async def _boom():
            raise HTTPException(status_code=418, detail="teapot")

        with TestClient(app) as client:
            r = client.get("/cohere/v2/boom")
        assert r.status_code == 418
        body = r.json()
        assert body == {"message": "teapot"}

    def test_non_cohere_path_is_not_translated(self):
        app = _build_app_with_vllm_handlers(handler=None)

        @app.get("/v1/other")
        async def _other():
            raise HTTPException(status_code=400, detail="nope")

        with TestClient(app) as client:
            r = client.get("/v1/other")
        assert r.status_code == HTTPStatus.BAD_REQUEST
        body = r.json()
        # Non-cohere paths keep the vLLM ``ErrorResponse`` shape.
        assert "error" in body
        assert body["error"]["message"] == "nope"

    def test_streaming_response_passes_through(self):
        # SSE responses have content-type text/event-stream; the
        # middleware must never buffer these (which would break
        # streaming) even though they're on ``/cohere/*``.
        async def _gen() -> AsyncGenerator[str, None]:
            yield 'data: {"type":"message-start"}\n\n'
            yield "data: [DONE]\n\n"

        app = _build_app_with_vllm_handlers(handler=_Handler(_gen()))
        with TestClient(app) as client:
            r = client.post(
                "/cohere/v2/chat",
                json={**_minimal_request_body(), "stream": True},
            )
        assert r.status_code == HTTPStatus.OK
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "message-start" in r.text
        assert r.text.rstrip().endswith("[DONE]")

    def test_already_cohere_shaped_body_passes_through(self):
        # When the handler returns an ``ErrorResponse`` the route
        # itself translates it to ``CohereError``; the middleware sees
        # the ``CohereError`` shape and must leave it alone.
        err = ErrorResponse(
            error=ErrorInfo(message="already cohere", type="Bad Request", code=400)
        )
        app = _build_app_with_vllm_handlers(handler=_Handler(err))
        with TestClient(app) as client:
            r = client.post("/cohere/v2/chat", json=_minimal_request_body())
        assert r.status_code == HTTPStatus.BAD_REQUEST
        body = r.json()
        # No ``error`` wrapper: the route already emitted the wire shape.
        assert body == {"message": "already cohere"}

    def test_request_id_preserved_in_translated_body(self):
        # Client-provided ``X-Request-Id`` should be echoed as
        # ``CohereError.id`` so callers can correlate failures.
        app = _build_app_with_vllm_handlers(handler=None)
        with TestClient(app) as client:
            r = client.post(
                "/cohere/v2/chat",
                json={"messages": [], "model": ""},
                headers={"X-Request-Id": "req-abc"},
            )
        assert r.status_code == HTTPStatus.BAD_REQUEST
        body = r.json()
        assert body.get("id") == 