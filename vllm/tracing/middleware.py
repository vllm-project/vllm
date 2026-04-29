# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASGI middleware that extracts W3C trace context from HTTP request headers.

When OpenTelemetry is available and tracing is configured, this middleware
reads ``traceparent`` / ``tracestate`` headers from each incoming request
and attaches the extracted context to the current OTel context for the
duration of the request.  This ensures that any spans created by route
handlers (via ``@instrument`` or manual ``tracer.start_span``) are
automatically parented under the caller's trace.

If OTel is not installed or no trace headers are present, the middleware
is a transparent passthrough with negligible overhead.
"""

from collections.abc import Awaitable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.tracing.otel import is_otel_available

if is_otel_available():
    from opentelemetry import context as otel_context
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _propagator = TraceContextTextMapPropagator()


class TraceContextMiddleware:
    """Pure ASGI middleware that propagates W3C trace context from headers.

    Must be added **before** (i.e. wrapping) any middleware or route handler
    that creates OTel spans so they inherit the correct parent context.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> Awaitable[None]:
        if scope["type"] != "http" or not is_otel_available():
            return self.app(scope, receive, send)

        headers = Headers(scope=scope)

        # Fast-path: skip extraction if no trace headers present
        if "traceparent" not in headers:
            return self.app(scope, receive, send)

        # Extract W3C trace context from request headers
        ctx = _propagator.extract(headers)

        # Attach the extracted context and run the downstream app within it
        token = otel_context.attach(ctx)

        async def _run() -> None:
            try:
                await self.app(scope, receive, send)
            finally:
                otel_context.detach(token)

        return _run()
