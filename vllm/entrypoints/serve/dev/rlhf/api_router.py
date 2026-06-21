# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine import PauseMode

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


# ---------------------------------------------------------------------------
# Weight version state — lives on app.state, updated by finish_weight_update.
#
# Design (improvement over sglang):
#   weight_gen:   monotonic int, auto-incremented on every finish_weight_update.
#                 Used for off-policy detection (staleness = current - sample).
#   weight_label: optional client-set string (e.g. "step-1500").
#                 Purely informational, not used for ordering.
#
# Both are updated AFTER the engine client confirms finish_weight_update
# succeeded, so there is no window where gen is bumped but weights are stale.
# ---------------------------------------------------------------------------


class _WeightVersionState:
    """Thread-safe weight version tracker.

    Attached to ``app.state.weight_version`` at router registration time.

    NOTE: This state is per-process. With ``--data-parallel-size > 1``,
    each API server process maintains its own gen counter. Clients that
    use DP should either pin to one process or use weight_label for
    cross-process identity.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._gen: int = 0
        self._label: str = ""

    def bump(self, label: str | None = None) -> int:
        """Increment gen and optionally set label. Returns the new gen."""
        with self._lock:
            self._gen += 1
            if label is not None:
                self._label = label
            return self._gen

    def set_label(self, label: str) -> None:
        with self._lock:
            self._label = label

    def get(self) -> dict:
        with self._lock:
            return {
                "weight_gen": self._gen,
                "weight_label": self._label,
            }


router = APIRouter()


# ───────────────────────────────────────────────────────────────────────
# Pause / resume
# ───────────────────────────────────────────────────────────────────────


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


# ───────────────────────────────────────────────────────────────────────
# Weight transfer protocol
# ───────────────────────────────────────────────────────────────────────


@router.post("/init_weight_transfer_engine")
async def init_weight_transfer_engine(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904
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
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904
    is_checkpoint_format = body.get("is_checkpoint_format", True)
    await engine_client(raw_request).start_weight_update(
        is_checkpoint_format=is_checkpoint_format
    )
    return JSONResponse(content={"message": "Weight update started"})


@router.post("/update_weights")
async def update_weights(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904
    update_info = body.get("update_info")
    if update_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'update_info' in request body",
        )
    await engine_client(raw_request).update_weights(
        request=WeightTransferUpdateRequest(update_info=update_info)
    )
    return JSONResponse(content={"message": "Weights updated"})


@router.post("/finish_weight_update")
async def finish_weight_update(raw_request: Request):
    """Finish the current weight update session and bump weight_gen.

    The weight_gen counter is incremented AFTER the engine confirms the
    update is applied. An optional ``weight_label`` in the request body
    is stored alongside the gen for human-readable identification.
    """
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, RuntimeError):
        # RuntimeError: Starlette raises this if the body was already consumed.
        body = {}

    await engine_client(raw_request).finish_weight_update()

    # Bump version after engine confirms success.
    wv: _WeightVersionState = raw_request.app.state.weight_version
    label = body.get("weight_label")
    new_gen = wv.bump(label=label)

    return JSONResponse(content={
        "message": "Weight update finished",
        "weight_gen": new_gen,
        "weight_label": label if label is not None else wv.get()["weight_label"],
    })


# ───────────────────────────────────────────────────────────────────────
# Weight version info
# ───────────────────────────────────────────────────────────────────────


@router.get("/weight_info")
async def weight_info(raw_request: Request) -> JSONResponse:
    """Return current weight version info.

    Response::

        {
            "weight_gen": 3,          // monotonic int, bumped by finish_weight_update
            "weight_label": "step-1500"  // client-set label, "" if never set
        }
    """
    wv: _WeightVersionState = raw_request.app.state.weight_version
    return JSONResponse(content=wv.get())


@router.post("/update_weight_label")
async def update_weight_label(raw_request: Request) -> JSONResponse:
    """Set the weight_label without changing weight_gen.

    Useful for tagging weights loaded from disk or external sync where
    finish_weight_update was not called through the HTTP protocol.

    Body::

        {"weight_label": "step-1500"}
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904

    label = body.get("weight_label")
    if label is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'weight_label' in request body",
        )

    wv: _WeightVersionState = raw_request.app.state.weight_version
    wv.set_label(str(label))

    return JSONResponse(content=wv.get())


# ───────────────────────────────────────────────────────────────────────
# World size
# ───────────────────────────────────────────────────────────────────────


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


# ───────────────────────────────────────────────────────────────────────
# Registration
# ───────────────────────────────────────────────────────────────────────


def attach_router(app: FastAPI):
    # Initialize weight version state on the app.
    if not hasattr(app.state, "weight_version"):
        app.state.weight_version = _WeightVersionState()
    app.include_router(router)
