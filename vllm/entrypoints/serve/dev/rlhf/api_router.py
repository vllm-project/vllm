# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine import PauseMode

from .metrics import (
    rl_weight_gen,
    rl_weight_update_active,
    rl_weight_update_duration_seconds,
    rl_weight_update_total,
)
from .rl_state_machine import RLStateMachineState

_ENGINE_IDX = "0"


logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


class _WeightVersionState:
    """Track the generation and label of the active model weights."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen = 0
        self._label = ""
        self._update_start: float | None = None

    def mark_start(self) -> None:
        with self._lock:
            self._update_start = time.perf_counter()

    def bump(self, label: str | None = None) -> tuple[int, float | None]:
        with self._lock:
            self._gen += 1
            if label is not None:
                self._label = label
            elapsed = (
                time.perf_counter() - self._update_start
                if self._update_start is not None
                else None
            )
            self._update_start = None
            return self._gen, elapsed

    def set_label(self, label: str) -> None:
        with self._lock:
            self._label = label

    def get(self) -> dict[str, str | int]:
        with self._lock:
            return {"weight_gen": self._gen, "weight_label": self._label}


@router.post("/pause")
async def pause_generation(
    raw_request: Request,
    mode: Annotated[PauseMode, Query()] = "abort",
    wait_for_inflight_requests: bool = Query(False),
    clear_cache: Annotated[bool, Query()] = True,
) -> JSONResponse:
    """Pause generation requests to allow weight updates.

    Args:
        mode: How to handle in-flight requests:
            - ``"abort"``: Abort all in-flight requests immediately (default).
            - ``"wait"``: Wait for in-flight requests to complete.
            - ``"keep"``: Freeze requests in queue; they resume on /resume.
        wait_for_inflight_requests: DEPRECATED. Use ``mode="wait"`` instead.
        clear_cache: DEPRECATED. Whether to clear KV/prefix caches after
            draining. Ignored when mode="keep".
    """

    engine = engine_client(raw_request)

    try:
        await engine.pause_generation(
            mode=mode,
            clear_cache=clear_cache,
            wait_for_inflight_requests=wait_for_inflight_requests,
        )
        return JSONResponse(
            content={"status": "paused"},
            status_code=HTTPStatus.OK.value,
        )

    except ValueError as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to pause generation")
        return JSONResponse(
            content={"error": f"Failed to pause generation: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/resume")
async def resume_generation(raw_request: Request) -> JSONResponse:
    """Resume generation after a pause."""

    engine = engine_client(raw_request)

    try:
        await engine.resume_generation()
        return JSONResponse(
            content={"status": "resumed"},
            status_code=HTTPStatus.OK.value,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to resume generation")
        return JSONResponse(
            content={"error": f"Failed to resume generation: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/abort_requests")
async def abort_requests(raw_request: Request) -> JSONResponse:
    """Abort in-flight requests without pausing the scheduler.

    Empty/missing ``request_ids`` aborts all in-flight requests.
    """

    engine = engine_client(raw_request)

    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format"
        ) from exc  # noqa: B904

    request_ids = body.get("request_ids")

    try:
        if request_ids:
            # Body ids are external (user-supplied) request ids.
            await engine.abort(request_ids)
        else:
            # The dev RL server runs AsyncLLM; abort everything it is tracking.
            # request_states is keyed by internal ids; parent_requests holds
            # parallel-sampling parents. Abort both as internal ids.
            from vllm.v1.engine.async_llm import AsyncLLM

            assert isinstance(engine, AsyncLLM)
            op = engine.output_processor
            request_ids = [
                *op.request_states.keys(),
                *op.parent_requests.keys(),
            ]
            await engine.abort(request_ids, internal=True)
        return JSONResponse(
            content={"status": "aborted", "aborted": len(request_ids)},
            status_code=HTTPStatus.OK.value,
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to abort requests")
        return JSONResponse(
            content={"error": f"Failed to abort requests: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.get("/is_paused")
async def is_paused(raw_request: Request) -> JSONResponse:
    """Return the current pause status."""

    engine = engine_client(raw_request)

    try:
        paused = await engine.is_paused()
    except Exception as err:  # pragma: no cover - defensive
        logger.exception("Failed to fetch pause status")
        return JSONResponse(
            content={"error": f"Failed to fetch pause status: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    return JSONResponse(content={"is_paused": paused})


@router.post("/init_weight_transfer_engine")
async def init_weight_transfer_engine(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format"
        ) from exc  # noqa: B904
    init_info = body.get("init_info")
    if init_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'init_info' in request body",
        )
    await engine_client(raw_request).init_weight_transfer_engine(
        WeightTransferInitRequest(init_info=init_info)
    )
    return JSONResponse(content={"message": "Weight transfer initialized"})


@router.post("/start_weight_update")
async def start_weight_update(raw_request: Request):
    sm: RLStateMachineState = raw_request.app.state.rl_state
    if sm.is_updating:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=(
                "start_weight_update called while a weight update is already "
                "in progress. Call finish_weight_update first."
            ),
        )

    await engine_client(raw_request).start_weight_update()
    await sm.on_start_weight_update()

    weight_state: _WeightVersionState = raw_request.app.state.weight_version
    weight_state.mark_start()
    rl_weight_update_active.labels(engine=_ENGINE_IDX).set(1)
    return JSONResponse(content={"message": "Weight update started"})


@router.post("/start_draft_weight_update")
async def start_draft_weight_update(raw_request: Request):
    await engine_client(raw_request).start_draft_weight_update()
    return JSONResponse(content={"message": "Draft weight update started"})


@router.post("/update_weights")
async def update_weights(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format"
        ) from exc  # noqa: B904
    update_info = body.get("update_info")
    if update_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'update_info' in request body",
        )

    state: RLStateMachineState = raw_request.app.state.rl_state
    try:
        await state.on_update_weights()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=str(exc),
        ) from exc

    await engine_client(raw_request).update_weights(
        request=WeightTransferUpdateRequest(update_info=update_info)
    )
    return JSONResponse(content={"message": "Weights updated"})


@router.post("/finish_weight_update")
async def finish_weight_update(raw_request: Request):
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, RuntimeError):
        body = {}

    weight_version = body.get("weight_version")
    weight_label = body.get("weight_label")

    state: RLStateMachineState = raw_request.app.state.rl_state
    if not state.is_updating:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=(
                "finish_weight_update called without a preceding "
                "start_weight_update."
            ),
        )

    await engine_client(raw_request).finish_weight_update(weight_version)

    weight_state: _WeightVersionState = raw_request.app.state.weight_version
    try:
        await state.on_finish_weight_update()
        new_gen, elapsed = weight_state.bump(label=weight_label)
        rl_weight_update_total.labels(engine=_ENGINE_IDX).inc()
        rl_weight_gen.labels(engine=_ENGINE_IDX).set(new_gen)
        if elapsed is not None:
            rl_weight_update_duration_seconds.labels(
                engine=_ENGINE_IDX
            ).observe(elapsed)
    finally:
        rl_weight_update_active.labels(engine=_ENGINE_IDX).set(0)

    return JSONResponse(
        content={
            "message": "Weight update finished",
            "weight_gen": new_gen,
            "weight_label": weight_state.get()["weight_label"],
        }
    )


@router.post("/update_weight_version")
async def update_weight_version(
    raw_request: Request,
    new_version: Annotated[str, Body(embed=True)],
):
    await engine_client(raw_request).update_weight_version(new_version)
    return JSONResponse(content={"success": True, "new_version": new_version})


@router.post("/update_weight_label")
async def update_weight_label(raw_request: Request) -> JSONResponse:
    """Set the human-readable weight label without changing weight_gen."""
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from exc

    label = body.get("weight_label")
    if label is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'weight_label' in request body",
        )

    weight_state: _WeightVersionState = raw_request.app.state.weight_version
    weight_state.set_label(str(label))
    return JSONResponse(content=weight_state.get())


@router.get("/weight_info")
async def weight_info(raw_request: Request):
    weight_version = await engine_client(raw_request).get_weight_version()
    return JSONResponse(
        content={
            "weight_version": weight_version,
            **raw_request.app.state.weight_version.get(),
        }
    )


@router.get("/get_world_size")
async def get_world_size(
    raw_request: Request,
    include_dp: bool = Query(True),
):
    """Get the world size from the parallel config.

    Args:
        include_dp: If True (default), returns the world size including
            data parallelism (TP * PP * DP). If False, returns the world
            size without data parallelism (TP * PP).
    """
    parallel_config = engine_client(raw_request).vllm_config.parallel_config
    if include_dp:
        world_size = parallel_config.world_size_across_dp
    else:
        world_size = parallel_config.world_size
    return JSONResponse(content={"world_size": world_size})


def attach_router(app: FastAPI):
    if not hasattr(app.state, "weight_version"):
        app.state.weight_version = _WeightVersionState()
    if not hasattr(app.state, "rl_state"):
        app.state.rl_state = RLStateMachineState()

    # Seed labeled children so the metrics are visible immediately at startup.
    rl_weight_gen.labels(engine=_ENGINE_IDX).set(0)
    rl_weight_update_active.labels(engine=_ENGINE_IDX).set(0)
    rl_weight_update_total.labels(engine=_ENGINE_IDX)
    rl_weight_update_duration_seconds.labels(engine=_ENGINE_IDX)

    app.include_router(router)

