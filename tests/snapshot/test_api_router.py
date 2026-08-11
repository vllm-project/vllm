# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException, Request

from vllm.entrypoints.serve.dev.sleep.api_router import (
    _run_exclusive_snapshot_operation,
    _run_snapshot_lifecycle,
    is_sleeping,
)
from vllm.entrypoints.serve.instrumentator.health import ready
from vllm.snapshot.middleware import EngineSnapshotGate


def test_snapshot_lifecycle_finishes_before_request_cancellation():
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        completed = False

        async def operation():
            nonlocal completed
            started.set()
            await release.wait()
            completed = True
            return "done"

        task = asyncio.create_task(_run_snapshot_lifecycle(operation()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed

    asyncio.run(run())


def test_snapshot_operations_reject_concurrent_sleep_or_wake():
    async def run():
        gate = EngineSnapshotGate()
        started = asyncio.Event()
        release = asyncio.Event()
        request = cast(
            Request,
            SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(engine_snapshot_gate=gate))
            ),
        )

        async def operation():
            started.set()
            await release.wait()

        first = asyncio.create_task(
            _run_exclusive_snapshot_operation(request, operation)
        )
        await started.wait()

        with pytest.raises(HTTPException) as exc_info:
            await _run_exclusive_snapshot_operation(request, operation)
        assert exc_info.value.status_code == 409

        release.set()
        await first

    asyncio.run(run())


def test_ready_is_healthy_for_render_only_server():
    async def run():
        request = cast(
            Request,
            SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(engine_client=None))
            ),
        )
        response = await ready(request)
        assert response.status_code == 200

    asyncio.run(run())


def _snapshot_status_request(state: str) -> Request:
    client = SimpleNamespace(request=lambda command, payload=None: {"state": state})
    return cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(engine_snapshot_client=client))
        ),
    )


def test_is_sleeping_reports_failed_state_as_not_sleeping():
    async def run():
        response = await is_sleeping(_snapshot_status_request("FAILED"))
        assert json.loads(response.body) == {
            "is_sleeping": False,
            "snapshot_state": "FAILED",
        }

    asyncio.run(run())


def test_is_sleeping_reports_hibernated_state_as_level3_sleep():
    async def run():
        response = await is_sleeping(_snapshot_status_request("HIBERNATED"))
        assert json.loads(response.body) == {
            "is_sleeping": True,
            "level": 3,
            "snapshot_state": "HIBERNATED",
        }

    asyncio.run(run())
