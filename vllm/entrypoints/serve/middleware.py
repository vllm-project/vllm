# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Global variable to track server unavailability (scaling, draining, etc.)
_server_unavailable = False


def is_server_unavailable() -> bool:
    return _server_unavailable


def set_server_unavailable(value: bool) -> None:
    global _server_unavailable
    _server_unavailable = value


class ScalingMiddleware:
    """
    Middleware that checks if the server is currently unavailable
    (e.g., scaling or draining) and returns a 503 Service Unavailable.

    This middleware applies to all HTTP requests and prevents
    processing when the server is unavailable.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        if is_server_unavailable():
            response = JSONResponse(
                content={"error": "Server is unavailable. Please try again later."},
                status_code=503,
            )
            return response(scope, receive, send)

        return self.app(scope, receive, send)
