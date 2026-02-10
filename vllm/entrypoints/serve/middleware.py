# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# tracks whether server is rejecting new requests (during scaling, shutdown, etc.)
_rejecting_requests = False

# paths that should remain accessible during drain (liveness + observability)
_EXEMPT_PATHS = {"/live", "/metrics"}


def is_rejecting_requests() -> bool:
    return _rejecting_requests


def set_rejecting_requests(value: bool) -> None:
    global _rejecting_requests
    _rejecting_requests = value


class ServiceUnavailableMiddleware:
    """
    Middleware that checks if the server is currently unavailable
    (e.g., scaling or draining) and returns a 503 Service Unavailable.

    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if is_rejecting_requests() and scope.get("path", "") not in _EXEMPT_PATHS:
            response = JSONResponse(
                content={"error": "Server is unavailable. Please try again later."},
                status_code=503,
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except asyncio.CancelledError:
            if not is_rejecting_requests():
                raise
            try:
                response = JSONResponse(
                    content={"error": "Server is shutting down."},
                    status_code=503,
                )
                await response(scope, receive, send)
            except (Exception, asyncio.CancelledError):
                pass
