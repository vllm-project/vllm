# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASGI middleware for WebSocket Prometheus metrics.

Modeled after prometheus-fastapi-instrumentator, this middleware
transparently instruments WebSocket endpoints with standard metrics
without requiring changes to handler code.

NOTE: This module intentionally has zero vllm imports so that it can
be extracted into a standalone package (similar to
prometheus-fastapi-instrumentator) in the future. Please keep it that way.
"""

import time
from collections.abc import Awaitable

from prometheus_client import Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Standard WebSocket metric names (not vllm-specific, following
# the same convention as prometheus-fastapi-instrumentator).
_active_sessions = Gauge(
    name="vllm:websocket_connections_active",
    documentation="Number of currently active WebSocket connections.",
    multiprocess_mode="livesum",
)

_total_sessions = Counter(
    name="vllm:websocket_connections_total",
    documentation="Total number of WebSocket connections.",
)

_session_duration = Histogram(
    name="vllm:websocket_connection_duration_seconds",
    documentation="Duration of WebSocket connections in seconds.",
    buckets=[0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800],
)


class WebSocketMetricsMiddleware:
    """Pure ASGI middleware that instruments WebSocket connections.

    Tracks active connections (gauge), total connections (counter),
    and connection duration (histogram) for all WebSocket endpoints.

    Usage::

        app.add_middleware(WebSocketMetricsMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "websocket":
            return self.app(scope, receive, send)

        return self._handle_websocket(scope, receive, send)

    async def _handle_websocket(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        start_time: float | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal start_time
            if message["type"] == "websocket.accept":
                start_time = time.monotonic()
                _active_sessions.inc()
                _total_sessions.inc()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if start_time is not None:
                _active_sessions.dec()
                _session_duration.observe(time.monotonic() - start_time)
