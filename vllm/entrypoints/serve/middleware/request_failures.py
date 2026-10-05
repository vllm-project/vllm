# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus counter for requests that fail in the API server.

``vllm:request_success`` only covers requests that reached the engine, so
failures in the API server are otherwise only visible in the logs.
"""

from collections.abc import Awaitable
from enum import Enum
from typing import cast

from fastapi import Request
from prometheus_client import REGISTRY, Counter
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.entrypoints.serve.instrumentator.metrics import UNINSTRUMENTED_HANDLERS

_GENERATION_STARTED = "generation_started"

_REQUEST_FAILURES = "vllm:request_failure"

_model_name: str | None = None
_request_failures: Counter | None = None


class RequestFailureStage(Enum):
    # Before generation started, e.g. validation, rendering or admission.
    INPUT_PROCESSING = "input_processing"
    # After the stream started, so the HTTP status was already sent as 200.
    STREAMING = "streaming"


def init_request_failure_metrics(*, model_name: str) -> None:
    """Register the counter.

    Must run after the engine is created, since that unregisters all
    existing ``vllm:`` collectors.
    """
    global _model_name, _request_failures
    _model_name = model_name
    try:
        _request_failures = Counter(
            name=_REQUEST_FAILURES,
            documentation=(
                "Number of requests that failed in the API server, by the stage "
                "the failure surfaced in and the HTTP status code of the error."
            ),
            labelnames=["model_name", "stage", "code"],
        )
    except ValueError:
        _request_failures = cast(
            Counter, REGISTRY._names_to_collectors[_REQUEST_FAILURES]
        )


def record_request_failure(stage: RequestFailureStage, status_code: int) -> None:
    if _request_failures is None:
        return
    _request_failures.labels(
        model_name=_model_name, stage=stage.value, code=str(status_code)
    ).inc()


def mark_generation_started(raw_request: Request | None) -> None:
    """Mark that input processing is done and the response is being generated."""
    if raw_request is not None:
        setattr(raw_request.state, _GENERATION_STARTED, True)


class RequestFailureMetricsMiddleware:
    """Pure ASGI middleware that counts error responses of API routes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        return self._handle_http(scope, receive, send)

    async def _handle_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                if message["status"] >= 400:
                    self._record(scope, message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except ClientDisconnect:
            raise
        except Exception:
            if response_started:
                # The response body broke off, e.g. a stream generator raised.
                record_request_failure(RequestFailureStage.STREAMING, 500)
            else:
                # Unhandled errors become a 500 in ServerErrorMiddleware,
                # which runs outside of this middleware.
                self._record(scope, 500)
            raise

    @staticmethod
    def _record(scope: Scope, status_code: int) -> None:
        route = scope.get("route")
        # Unmatched paths and probes are not requests for the model.
        if route is None or route.path in UNINSTRUMENTED_HANDLERS:
            return
        if scope.get("state", {}).get(_GENERATION_STARTED):
            return
        record_request_failure(RequestFailureStage.INPUT_PROCESSING, status_code)
