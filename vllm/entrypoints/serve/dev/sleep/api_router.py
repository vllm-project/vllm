# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.snapshot.protocol import SnapshotControlClient
from vllm.v1.engine import PauseMode

logger = init_logger(__name__)
_T = TypeVar("_T")


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def snapshot_client(request: Request) -> SnapshotControlClient | None:
    return getattr(request.app.state, "engine_snapshot_client", None)


router = APIRouter()


async def _attach_engine(
    raw_request: Request,
    client: SnapshotControlClient,
    ticket: Mapping[str, Any],
    resource_policy: dict[str, str],
    *,
    rollback: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    started = time.monotonic()
    timings: dict[str, float] = {}
    engine = engine_client(raw_request)
    phase_started = time.monotonic()
    await engine.snapshot_wait_for_attach(
        ticket["nonce"],
        ticket["generation"],
        ticket["snapshot_id"],
        ticket["config_hash"],
        ticket["root_pid"],
    )
    timings["attach_wait_seconds"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    await asyncio.to_thread(
        client.request,
        "confirm_attach",
        {
            field: ticket[field]
            for field in (
                "nonce",
                "generation",
                "snapshot_id",
                "config_hash",
                "root_pid",
            )
        },
    )
    timings["manager_confirm_attach_seconds"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    worker_timings = await engine.checkpoint_restore(resource_policy)
    timings["worker_restore_seconds"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    await engine.resume_generation()
    timings["resume_generation_seconds"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    await engine.check_health()
    timings["health_check_seconds"] = time.monotonic() - phase_started
    command = "complete_capture_rollback" if rollback else "complete_restore"
    phase_started = time.monotonic()
    await asyncio.to_thread(client.request, command)
    timings["manager_complete_seconds"] = time.monotonic() - phase_started
    timings["total_seconds"] = time.monotonic() - started
    return timings, worker_timings


async def _fail_capture(client: SnapshotControlClient, error: BaseException) -> None:
    status = await asyncio.to_thread(client.request, "status")
    if status["state"] == "FAILED":
        return
    if status["state"] not in ("DRAINING", "ATTACHING", "VERIFYING"):
        raise RuntimeError(f"cannot fail capture from state {status['state']}")
    await asyncio.to_thread(client.request, "fail_capture", {"error": str(error)})


async def _run_snapshot_lifecycle(operation: Awaitable[_T]) -> _T:
    """Finish a snapshot lifecycle operation before honoring cancellation."""
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Snapshot lifecycle failed after request cancellation")
        raise


async def _run_exclusive_snapshot_operation(
    raw_request: Request,
    operation: Callable[[], Awaitable[_T]],
) -> _T:
    gate = raw_request.app.state.engine_snapshot_gate
    if not await gate.begin_operation():
        raise HTTPException(409, "Another engine sleep or wake operation is running")
    try:
        return await operation()
    finally:
        await gate.end_operation()


@router.post("/sleep")
async def sleep(raw_request: Request) -> Response:
    level = raw_request.query_params.get("level", "1")
    mode_value = raw_request.query_params.get("mode", "abort")
    if mode_value not in ("abort", "wait", "keep"):
        raise HTTPException(400, "Sleep mode must be abort, wait, or keep")
    mode = cast(PauseMode, mode_value)
    try:
        level_number = int(level)
    except ValueError as exc:
        raise HTTPException(400, "Sleep level must be an integer") from exc
    if level_number not in (1, 2, 3):
        raise HTTPException(400, "Sleep level must be 1, 2, or 3")
    client = snapshot_client(raw_request)
    if client is not None:
        return await _run_snapshot_lifecycle(
            _run_exclusive_snapshot_operation(
                raw_request,
                lambda: _sleep_with_snapshot(raw_request, client, level_number, mode),
            )
        )
    if level_number == 3:
        raise HTTPException(400, "Sleep Mode level 3 is not enabled")
    await engine_client(raw_request).sleep(level_number, mode)
    # FIXME: in v0 with frontend multiprocessing, the sleep command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


async def _sleep_with_snapshot(
    raw_request: Request,
    client: SnapshotControlClient,
    level: int,
    mode: PauseMode,
) -> Response:
    if level == 3:
        return await _sleep_level3(raw_request, mode)
    status = await asyncio.to_thread(client.request, "status")
    if status["state"] != "READY":
        raise HTTPException(409, f"Engine snapshot state is {status['state']}")
    await engine_client(raw_request).sleep(level, mode)
    return Response(status_code=200)


async def _sleep_level3(raw_request: Request, mode: PauseMode) -> JSONResponse:
    started = time.monotonic()
    timings: dict[str, float] = {}
    client = snapshot_client(raw_request)
    if client is None:
        raise HTTPException(400, "Sleep Mode level 3 is not enabled")
    status = await asyncio.to_thread(client.request, "status")
    if status["state"] != "READY":
        raise HTTPException(409, f"Engine snapshot state is {status['state']}")
    engine = engine_client(raw_request)
    resource_policy = status["resource_policy"]
    gate = raw_request.app.state.engine_snapshot_gate
    phase_started = time.monotonic()
    active_requests = await gate.close()
    timings["gate_close_seconds"] = time.monotonic() - phase_started
    if active_requests:
        await gate.open()
        raise HTTPException(409, f"Engine has {active_requests} active API request(s)")
    phase_started = time.monotonic()
    initially_idle = await engine.is_idle()
    timings["initial_idle_check_seconds"] = time.monotonic() - phase_started
    if not initially_idle:
        await gate.open()
        raise HTTPException(409, "Engine has in-flight requests")
    manager_prepared = False
    checkpoint_prepare_started = False
    paused = False
    ticket: Mapping[str, Any] | None = None
    detached = False
    try:
        phase_started = time.monotonic()
        prepared_result = await asyncio.to_thread(client.request, "prepare_capture")
        timings["manager_prepare_capture_seconds"] = time.monotonic() - phase_started
        manager_prepared = True
        ticket = prepared_result["ticket"]
        phase_started = time.monotonic()
        still_idle = await engine.is_idle()
        timings["idle_recheck_seconds"] = time.monotonic() - phase_started
        if not still_idle:
            raise HTTPException(409, "Engine became busy during capture")
        phase_started = time.monotonic()
        await engine.pause_generation(
            mode=mode, clear_cache=resource_policy["kv"] == "discard"
        )
        timings["pause_generation_seconds"] = time.monotonic() - phase_started
        paused = True
        phase_started = time.monotonic()
        barrier_idle = await engine.is_idle()
        timings["idle_barrier_seconds"] = time.monotonic() - phase_started
        if not barrier_idle:
            raise HTTPException(409, "Engine did not reach the idle barrier")
        phase_started = time.monotonic()
        checkpoint_prepare_started = True
        worker_timings = await engine.checkpoint_prepare(resource_policy)
        timings["worker_prepare_seconds"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        await engine.snapshot_detach_io(
            ticket["nonce"],
            ticket["generation"],
            ticket["snapshot_id"],
            ticket["config_hash"],
            ticket["marker_path"],
            prepared_result["persistence"],
        )
        timings["zmq_detach_seconds"] = time.monotonic() - phase_started
        detached = True
        phase_started = time.monotonic()
        result = await asyncio.to_thread(client.request, "capture")
        timings["manager_capture_seconds"] = time.monotonic() - phase_started
        timings["total_seconds"] = time.monotonic() - started
        return JSONResponse(
            content=result | {"api_timings": timings, "worker_timings": worker_timings}
        )
    except (Exception, asyncio.CancelledError) as exc:
        rollback_error = None
        try:
            status = await asyncio.to_thread(client.request, "status")
            if detached and status["state"] == "ATTACHING":
                assert ticket is not None
                await _attach_engine(
                    raw_request,
                    client,
                    {**ticket, "root_pid": status["root_pid"]},
                    resource_policy,
                    rollback=True,
                )
            elif detached and status["state"] == "DRAINING":
                assert ticket is not None
                rollback = await asyncio.to_thread(client.request, "rollback_capture")
                await _attach_engine(
                    raw_request,
                    client,
                    rollback["ticket"],
                    resource_policy,
                    rollback=True,
                )
            elif detached and status["state"] != "HIBERNATED":
                raise RuntimeError(
                    f"cannot roll back capture from state {status['state']}"
                )
            elif checkpoint_prepare_started:
                await engine.checkpoint_abort()

            if not detached and paused:
                await engine.resume_generation()
            if manager_prepared:
                status = await asyncio.to_thread(client.request, "status")
                if status["state"] == "DRAINING":
                    await asyncio.to_thread(client.request, "abort_capture")
                elif status["state"] not in ("READY", "HIBERNATED"):
                    raise RuntimeError(
                        "capture rollback did not return snapshot manager to READY"
                    )
        except (Exception, asyncio.CancelledError) as cleanup_exc:
            rollback_error = cleanup_exc
            try:
                await _fail_capture(client, cleanup_exc)
            except (Exception, asyncio.CancelledError):
                logger.exception("Failed to mark snapshot operation as failed")

        if rollback_error is None:
            if not (detached and status["state"] == "HIBERNATED"):
                await gate.open()
            raise
        raise RuntimeError(f"{exc}; capture rollback failed: {rollback_error}") from exc


async def _wake_level3(
    raw_request: Request,
    client: SnapshotControlClient,
    status: Mapping[str, Any],
) -> JSONResponse:
    started = time.monotonic()
    timings: dict[str, float] = {}
    phase_started = time.monotonic()
    result = await asyncio.to_thread(client.request, "restore")
    timings["manager_restore_seconds"] = time.monotonic() - phase_started
    try:
        provider_result = result["provider_result"]
        manager_timings = result["manager_timings"]
        attach_timings, worker_timings = await _attach_engine(
            raw_request,
            client,
            result["ticket"],
            status["resource_policy"],
        )
        result = await asyncio.to_thread(client.request, "status")
        timings.update(
            {
                "attach_total_seconds" if name == "total_seconds" else name: seconds
                for name, seconds in attach_timings.items()
            }
        )
        phase_started = time.monotonic()
        await raw_request.app.state.engine_snapshot_gate.open()
        timings["gate_open_seconds"] = time.monotonic() - phase_started
    except (Exception, asyncio.CancelledError) as exc:
        await asyncio.to_thread(client.request, "fail_restore", {"error": str(exc)})
        raise
    timings["total_seconds"] = time.monotonic() - started
    return JSONResponse(
        content=result
        | {
            "provider_result": provider_result,
            "manager_timings": manager_timings,
            "api_timings": timings,
            "worker_timings": worker_timings,
        }
    )


@router.post("/wake_up")
async def wake_up(raw_request: Request) -> Response:
    tags: list[str] | None = raw_request.query_params.getlist("tags") or None
    client = snapshot_client(raw_request)
    if client is not None:
        return await _run_snapshot_lifecycle(
            _run_exclusive_snapshot_operation(
                raw_request,
                lambda: _wake_with_snapshot(raw_request, client, tags),
            )
        )
    logger.info("wake up the engine with tags: %s", tags)
    await engine_client(raw_request).wake_up(tags)
    # FIXME: in v0 with frontend multiprocessing, the wake-up command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


async def _wake_with_snapshot(
    raw_request: Request,
    client: SnapshotControlClient,
    tags: list[str] | None,
) -> Response:
    status = await asyncio.to_thread(client.request, "status")
    if status["state"] == "HIBERNATED":
        if tags is not None:
            raise HTTPException(400, "tags are not supported for level 3 wake")
        return await _wake_level3(raw_request, client, status)
    if status["state"] != "READY":
        raise HTTPException(409, f"Engine snapshot state is {status['state']}")
    logger.info("wake up the engine with tags: %s", tags)
    await engine_client(raw_request).wake_up(tags)
    return Response(status_code=200)


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request) -> JSONResponse:
    client = snapshot_client(raw_request)
    if client is not None:
        status = await asyncio.to_thread(client.request, "status")
        if status["state"] == "FAILED":
            return JSONResponse(
                content={
                    "is_sleeping": False,
                    "snapshot_state": status["state"],
                }
            )
        if status["state"] != "READY":
            return JSONResponse(
                content={
                    "is_sleeping": True,
                    "level": 3,
                    "snapshot_state": status["state"],
                }
            )
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


@router.get("/snapshot/status")
async def snapshot_status(raw_request: Request) -> JSONResponse:
    client = snapshot_client(raw_request)
    if client is None:
        raise HTTPException(404, "Engine snapshots are not enabled")
    return JSONResponse(content=await asyncio.to_thread(client.request, "status"))


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
