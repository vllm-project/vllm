# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from starlette.types import ASGIApp, Receive, Scope, Send

# Endpoints that submit work to the engine. Requests to anything else
# (health probes, /metrics, /tokenize, dev endpoints, ...) neither reset
# the idle timer nor wake a sleeping engine.
_INFERENCE_PATH_PREFIXES = (
    "/v1/",
    "/classify",
    "/inference",
    "/invocations",
    "/pooling",
    "/rerank",
    "/score",
)


def is_inference_request(scope: Scope) -> bool:
    return scope.get("method") == "POST" and scope.get("path", "").startswith(
        _INFERENCE_PATH_PREFIXES
    )


class IdleSleepMiddleware:
    """Pure ASGI middleware tracking in-flight inference requests.

    Runs the full request/response cycle inside the in-flight window so
    streaming responses keep the engine awake until the last byte is sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_inference_request(scope):
            return await self.app(scope, receive, send)

        manager = getattr(scope["app"].state, "idle_sleep_manager", None)
        if manager is None:
            return await self.app(scope, receive, send)

        await manager.on_request_start()
        try:
            await self.app(scope, receive, send)
        finally:
            manager.on_request_end()
