# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class FaultToleranceMiddleware:
    """
    ASGI middleware that checks whether the engine has been faulted by
    the ClientSentinel and short-circuits HTTP requests with 503 while
    the sentinel is in the faulted state.

    The middleware looks up the EngineClient on the FastAPI app state
    (`scope["app"].state.engine_client`) and then tries a few locations
    where a ClientSentinel may be stored on concrete client implementations:
      - `engine_client.client_sentinel`
      - `engine_client.engine_core.client_sentinel`
      - `engine_client.resources.client_sentinel`

    The checks are defensive: missing attributes or errors during the
    probe will *not* block requests (middleware is fail-open) but will be
    logged at warning level.

    NOTE: Requests to the fault-tolerance management endpoints themselves
    (for example any path under "/fault_tolerance") should be allowed
    even when the sentinel is faulted so that operators can query or
    clear the fault. The middleware checks the request path and skips
    the 503 short-circuit for those paths.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # Paths that should bypass the faulted short-circuit even when the
        # sentinel is faulted. Keep as a tuple so it's easy to extend.
        self._allowed_prefixes = ("/fault_tolerance",)

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        # Only apply to HTTP requests.
        if scope.get("type") != "http":
            return self.app(scope, receive, send)

        # If the request is for a fault-tolerance management endpoint,
        # allow it through so operators can inspect/clear faults.
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

            sentinel = engine_client.engine_core.client_sentinel

            # If we found a sentinel, check its is_faulted event.
            if sentinel is not None and sentinel.is_faulted.is_set():
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
