# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from typing import cast

from fastapi import Request

from vllm.snapshot.middleware import (
    EngineSnapshotGate,
    EngineSnapshotMiddleware,
    snapshot_control_error_handler,
)
from vllm.snapshot.protocol import SnapshotControlError


def test_snapshot_control_errors_surface_code_and_conflict_status():
    request = cast(Request, None)
    conflict = snapshot_control_error_handler(
        request, SnapshotControlError("engine is busy", code="busy")
    )
    assert conflict.status_code == 409
    assert json.loads(conflict.body) == {"error": "engine is busy", "code": "busy"}

    internal = snapshot_control_error_handler(
        request, SnapshotControlError("capture failed")
    )
    assert internal.status_code == 500
    assert json.loads(internal.body) == {
        "error": "capture failed",
        "code": "snapshot_error",
    }


def test_gate_closes_without_request_race():
    async def run():
        gate = EngineSnapshotGate()
        assert await gate.enter()
        assert await gate.close() == 1
        assert not await gate.enter()
        await gate.leave()
        await gate.open()
        assert await gate.enter()
        await gate.leave()

    asyncio.run(run())


def test_gate_serializes_snapshot_operations():
    async def run():
        gate = EngineSnapshotGate()
        assert await gate.begin_operation()
        assert not await gate.begin_operation()
        await gate.end_operation()
        assert await gate.begin_operation()
        await gate.end_operation()

    asyncio.run(run())


def test_gate_counts_realtime_websocket():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def app(scope, receive, send):
            entered.set()
            await release.wait()

        gate = EngineSnapshotGate()
        middleware = EngineSnapshotMiddleware(app, gate)
        task = asyncio.create_task(
            middleware(
                {"type": "websocket", "path": "/v1/realtime"},
                None,
                None,
            )
        )
        await entered.wait()
        assert await gate.close() == 1
        release.set()
        await task

    asyncio.run(run())


def test_gate_rejects_new_realtime_websocket_when_closed():
    async def run():
        sent = []

        async def app(scope, receive, send):
            raise AssertionError("closed gate must not call the app")

        async def send(message):
            sent.append(message)

        gate = EngineSnapshotGate()
        await gate.close()
        middleware = EngineSnapshotMiddleware(app, gate)
        await middleware(
            {"type": "websocket", "path": "/v1/realtime"},
            None,
            send,
        )

        assert sent == [
            {
                "type": "websocket.close",
                "code": 1013,
                "reason": "engine unavailable",
            }
        ]

    asyncio.run(run())
