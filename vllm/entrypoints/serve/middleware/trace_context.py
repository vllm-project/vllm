# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send


class TraceContextMiddleware:
    """Bind incoming W3C trace context for request-scoped logging."""

    def __init__(self, app: ASGIApp) -> None:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        self.app = app
        self.trace = trace
        self.propagator = TraceContextTextMapPropagator()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (
            self.trace.get_current_span().get_span_context().is_valid
        ):
            await self.app(scope, receive, send)
            return

        context = self.propagator.extract(Headers(scope=scope))
        span = self.trace.get_current_span(context)
        with self.trace.use_span(span, end_on_exit=False):
            await self.app(scope, receive, send)
