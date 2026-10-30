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
    _sleep_level3,
    _wake_level3,
    is_sleeping,
    ready,
)
from vllm.snapshot.middleware import EngineSnapshotGate
from vllm.snapshot.protocol import SnapshotControlError


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
    client = SimpleNamespace(
        request=lambda command, payload=None, **kwargs: {"state": state}
    )
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


class _FakeControlClient:
    def __init__(
        self,
        statuses: list[str],
        *,
        capture_error: BaseException | None = None,
        restore_error: BaseException | None = None,
    ) -> None:
        self.statuses = list(statuses)
        self.capture_error = capture_error
        self.restore_error = restore_error
        self.calls: list[str] = []
        self.ticket = {
            "nonce": "nonce",
            "generation": 1,
            "snapshot_id": "snapshot",
            "config_hash": "config",
            "root_pid": 4242,
            "marker_path": "/tmp/marker",
        }

    def request(self, command, payload=None, *, timeout=None):
        self.calls.append(command)
        if command == "status":
            return {"state": self.statuses.pop(0), "root_pid": 4242}
        if command == "prepare_capture":
            return {"ticket": self.ticket, "persistence": "durable"}
        if command == "capture":
            if self.capture_error is not None:
                raise self.capture_error
            return {"state": "HIBERNATED"}
        if command == "rollback_capture":
            return {"ticket": self.ticket}
        if command == "confirm_attach":
            return {"state": "VERIFYING"}
        if command == "complete_capture_rollback":
            return {"state": "READY"}
        if command == "fail_capture":
            return {"state": "FAILED"}
        if command == "restore":
            if self.restore_error is not None:
                raise self.restore_error
            return {
                "provider_result": {},
                "manager_timings": {},
                "ticket": self.ticket,
            }
        if command == "fail_restore":
            return {"state": "FAILED"}
        raise AssertionError(f"unexpected control command: {command}")


class _FakeEngine:
    def __init__(
        self,
        *,
        fail_prepare: bool = False,
        fail_attach: bool = False,
    ) -> None:
        self.fail_prepare = fail_prepare
        self.fail_attach = fail_attach
        self.calls: list[str] = []

    async def is_idle(self) -> bool:
        self.calls.append("is_idle")
        return True

    async def pause_generation(self, *, mode, clear_cache) -> None:
        self.calls.append("pause_generation")
        assert clear_cache is True

    async def snapshot_prepare(self):
        self.calls.append("snapshot_prepare")
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        return {"total_seconds": 0.0}

    async def snapshot_detach_io(self, *args) -> None:
        self.calls.append("snapshot_detach_io")

    async def snapshot_wait_for_attach(self, *args) -> None:
        self.calls.append("snapshot_wait_for_attach")
        if self.fail_attach:
            raise RuntimeError("attach failed")

    async def snapshot_restore(self):
        self.calls.append("snapshot_restore")
        return {"total_seconds": 0.0}

    async def snapshot_abort(self):
        self.calls.append("snapshot_abort")
        return {"total_seconds": 0.0}

    async def resume_generation(self) -> None:
        self.calls.append("resume_generation")

    async def check_health(self) -> None:
        self.calls.append("check_health")


def _snapshot_request(client, engine, gate):
    return cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    engine_snapshot_client=client,
                    engine_client=engine,
                    engine_snapshot_gate=gate,
                )
            )
        ),
    )


def test_capture_rollback_from_draining_reattaches():
    async def run():
        client = _FakeControlClient(
            ["READY", "DRAINING", "READY"],
            capture_error=RuntimeError("capture failed"),
        )
        engine = _FakeEngine()
        gate = EngineSnapshotGate()
        request = _snapshot_request(client, engine, gate)

        with pytest.raises(RuntimeError, match="capture failed"):
            await _sleep_level3(request, "abort")

        assert "rollback_capture" in client.calls
        assert "confirm_attach" in client.calls
        assert "complete_capture_rollback" in client.calls
        assert "fail_capture" not in client.calls
        assert await gate.enter()
        await gate.leave()

    asyncio.run(run())


def test_capture_rollback_from_attaching_reattaches():
    async def run():
        client = _FakeControlClient(
            ["READY", "ATTACHING", "READY"],
            capture_error=RuntimeError("capture failed"),
        )
        engine = _FakeEngine()
        gate = EngineSnapshotGate()
        request = _snapshot_request(client, engine, gate)

        with pytest.raises(RuntimeError, match="capture failed"):
            await _sleep_level3(request, "abort")

        assert "rollback_capture" not in client.calls
        assert "confirm_attach" in client.calls
        assert "complete_capture_rollback" in client.calls
        assert "fail_capture" not in client.calls

    asyncio.run(run())


def test_capture_failure_before_detach_resumes():
    async def run():
        client = _FakeControlClient(["READY", "READY", "READY"])
        engine = _FakeEngine(fail_prepare=True)
        gate = EngineSnapshotGate()
        request = _snapshot_request(client, engine, gate)

        with pytest.raises(RuntimeError, match="prepare failed"):
            await _sleep_level3(request, "abort")

        assert "snapshot_abort" in engine.calls
        assert "resume_generation" in engine.calls
        assert "confirm_attach" not in client.calls
        assert "rollback_capture" not in client.calls
        assert "fail_capture" not in client.calls

    asyncio.run(run())


def test_capture_cleanup_failure_routes_to_fail_capture():
    async def run():
        client = _FakeControlClient(
            ["READY", "ATTACHING", "ATTACHING"],
            capture_error=RuntimeError("capture failed"),
        )
        engine = _FakeEngine(fail_attach=True)
        gate = EngineSnapshotGate()
        request = _snapshot_request(client, engine, gate)

        with pytest.raises(RuntimeError, match="capture rollback failed"):
            await _sleep_level3(request, "abort")

        assert "fail_capture" in client.calls
        assert not await gate.enter()

    asyncio.run(run())


def test_wake_restore_failure_routes_to_fail_restore():
    async def run():
        client = _FakeControlClient(
            ["ATTACHING"],
            restore_error=SnapshotControlError(
                "snapshot control connection failed", code="connection_error"
            ),
        )
        engine = _FakeEngine()
        gate = EngineSnapshotGate()
        request = _snapshot_request(client, engine, gate)

        with pytest.raises(SnapshotControlError):
            await _wake_level3(request, client)

        assert client.calls == ["restore", "status", "fail_restore"]

    asyncio.run(run())


def test_wake_attach_failure_routes_to_fail_restore():
    async def run():
        client = _FakeControlClient(["ATTACHING"])
        engine = _FakeEngine(fail_attach=True)
        gate = EngineSnapshotGate()
        request = _snapshot_request(client, engine, gate)

        with pytest.raises(RuntimeError, match="attach failed"):
            await _wake_level3(request, client)

        assert "fail_restore" in client.calls

    asyncio.run(run())
