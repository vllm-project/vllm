# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.snapshot.protocol import SnapshotControlError

_CONFLICT_CODES = frozenset({"busy", "invalid_state", "identity_mismatch"})


def snapshot_control_error_handler(
    _: Request, exc: SnapshotControlError
) -> JSONResponse:
    status_code = 409 if exc.code in _CONFLICT_CODES else 500
    return JSONResponse(
        status_code=status_code,
        content={"error": str(exc), "code": exc.code},
    )


class EngineSnapshotGate:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._accepting = True
        self._active_requests = 0
        self._operation_active = False

    async def enter(self) -> bool:
        async with self._lock:
            if not self._accepting:
                return False
            self._active_requests += 1
            return True

    async def leave(self) -> None:
        async with self._lock:
            self._active_requests -= 1
            if self._active_requests < 0:
                raise RuntimeError("snapshot gate request count became negative")

    async def close(self) -> int:
        async with self._lock:
            self._accepting = False
            return self._active_requests

    async def open(self) -> None:
        async with self._lock:
            self._accepting = True

    async def begin_operation(self) -> bool:
        async with self._lock:
            if self._operation_active:
                return False
            self._operation_active = True
            return True

    async def end_operation(self) -> None:
        async with self._lock:
            if not self._operation_active:
                raise RuntimeError("snapshot gate has no active operation")
            self._operation_active = False


class EngineSnapshotMiddleware:
    _allowed_paths = {
        "/health",
        "/is_sleeping",
        "/metrics",
        "/ready",
        "/sleep",
        "/snapshot/status",
        "/wake_up",
    }

    def __init__(self, app: ASGIApp, gate: EngineSnapshotGate) -> None:
        self.app = app
        self.gate = gate

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] not in ("http", "websocket")
            or scope["path"] in self._allowed_paths
        ):
            await self.app(scope, receive, send)
            return
        if not await self.gate.enter():
            if scope["type"] == "websocket":
                await send(
                    {
                        "type": "websocket.close",
                        "code": 1013,
                        "reason": "engine unavailable",
                    }
                )
                return
            response = JSONResponse(
                status_code=503,
                content={"error": "engine unavailable"},
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            await self.gate.leave()
