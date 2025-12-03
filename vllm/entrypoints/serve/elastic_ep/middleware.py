# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Global variable to track scaling state
_scaling_elastic_ep = False


def get_scaling_elastic_ep():
    return _scaling_elastic_ep


def set_scaling_elastic_ep(value):
    global _scaling_elastic_ep
    _scaling_elastic_ep = value


class ScalingMiddleware:
    """
    Middleware that checks if the model is currently scaling and
    returns a 503 Service Unavailable response if it is.

    This middleware applies to all HTTP requests and prevents
    processing when the model is in a scaling state.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        # Check global scaling state
        if get_scaling_elastic_ep():
            # Return 503 Service Unavailable response
            response = JSONResponse(
                content={
                    "error": "The model is currently scaling. Please try again later."
                },
                status_code=503,
            )
            return response(scope, receive, send)

        return self.app(scope, receive, send)
