# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .state import PDRoleState


class PDRoleMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        state: PDRoleState | None = getattr(scope["app"].state, "pd_role", None)
        if state is None or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"].removeprefix(scope.get("root_path", ""))
        if scope["method"] in ("GET", "HEAD", "OPTIONS") or path in (
            "/v1/pd_role",
            "/tokenize",
            "/detokenize",
        ):
            await self.app(scope, receive, send)
            return

        if path not in ("/v1/completions", "/v1/chat/completions"):
            await JSONResponse(
                {"error": "This endpoint is unsupported with runtime P/D switching"},
                status_code=409,
            )(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if (
            state.phase != "ready"
            or headers.get("x-vllm-pd-role") != state.role
            or headers.get("x-vllm-pd-epoch") != str(state.epoch)
        ):
            await JSONResponse(
                {"error": "P/D admission rejected", **state.status()},
                status_code=409,
            )(scope, receive, send)
            return

        # No await between checking the fence and registering admission.
        state.active_requests += 1
        state.idle.clear()
        try:
            await self.app(scope, receive, send)
        finally:
            state.active_requests -= 1
            if state.active_requests == 0:
                state.idle.set()
