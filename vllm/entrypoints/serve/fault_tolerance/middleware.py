# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class FaultToleranceMiddleware:
    """
    ASGI middleware that short-circuits ordinary HTTP requests with a 503
    response when the engine is in a faulted state.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # Allowed management endpoint prefixes: these paths are permitted even
        # if the service is marked faulted.
        self._allowed_prefixes = ("/fault_tolerance",)

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        # Only apply to HTTP requests; forward other types unchanged.
        if scope.get("type") != "http":
            return self.app(scope, receive, send)

        # Allow management endpoints through so faults can be inspected/cleared.
        path = scope.get("path", "") or ""
        for prefix in self._allowed_prefixes:
            if path.startswith(prefix):
                return self.app(scope, receive, send)

        try:
            app_obj = scope.get("app")
            if app_obj is None:
                return self.app(scope, receive, send)

            # Engine client is attached to FastAPI app.state.engine_client by
            # the server initialization code. Be defensive in case it's not set.
            engine_client = getattr(app_obj.state, "engine_client", None)
            if engine_client is None:
                return self.app(scope, receive, send)

            # Check engine is_faulted event.
            if engine_client.engine_core.is_faulted.is_set():
                response = JSONResponse(
                    content={
                        "error": "Service is in faulted state, cannot process requests."
                    },
                    status_code=503,
                )
                return response(scope, receive, send)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning("FaultToleranceMiddleware unexpected error: %s", e)
            # Fail-open: allow requests to proceed.

        return self.app(scope, receive, send)
