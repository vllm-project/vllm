# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
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

from .metrics import (
    rl_weight_gen,
    rl_weight_update_active,
    rl_weight_update_duration_seconds,
    rl_weight_update_total,
)
from .rl_state_machine import RLStateMachineState, require_update_active, require_update_inactive

logger = init_logger(__name__)

# Engine index label for Prometheus (always "0" for single-engine deployments).
_ENGINE_IDX = "0"


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
        self._update_start: float | None = None  # perf_counter at start_weight_update

    def mark_start(self) -> None:
        with self._lock:
            self._update_start = time.perf_counter()

    def bump(self, label: str | None = None) -> tuple[int, float | None]:
        """Increment gen and optionally set label. Returns (new_gen, elapsed_s)."""
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
    # Enforce state machine: mark update in-progress (raises 409 on double-start).
    sm: RLStateMachineState = raw_request.app.state.rl_state
    try:
        sm.on_start_weight_update()
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.CONFLICT.value, detail=str(exc)) from exc
    # Record start time for duration histogram; mark Prometheus metric active.
    wv: _WeightVersionState = raw_request.app.state.weight_version
    wv.mark_start()
    rl_weight_update_active.labels(engine=_ENGINE_IDX).set(1)
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
    # State machine: update_weights must follow start_weight_update.
    sm: RLStateMachineState = raw_request.app.state.rl_state
    try:
        sm.on_update_weights()
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.CONFLICT.value, detail=str(exc)) from exc
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

    # State machine: finish requires preceding start.
    sm: RLStateMachineState = raw_request.app.state.rl_state
    try:
        sm.on_finish_weight_update()
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.CONFLICT.value, detail=str(exc)) from exc

    await engine_client(raw_request).finish_weight_update()

    # Bump version after engine confirms success; collect elapsed time.
    wv: _WeightVersionState = raw_request.app.state.weight_version
    label = body.get("weight_label")
    new_gen, elapsed_s = wv.bump(label=label)

    # Update Prometheus metrics.
    rl_weight_update_total.labels(engine=_ENGINE_IDX).inc()
    rl_weight_gen.labels(engine=_ENGINE_IDX).set(new_gen)
    rl_weight_update_active.labels(engine=_ENGINE_IDX).set(0)
    if elapsed_s is not None:
        rl_weight_update_duration_seconds.labels(engine=_ENGINE_IDX).observe(elapsed_s)

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
# Weight checksum (snapshot / compare / checksum / reset)
# ───────────────────────────────────────────────────────────────────────


class _WeightCheckerState:
    """Stores the last SHA-256 weight snapshot for compare operations.

    Thread-safe because all HTTP handlers run in the same asyncio event loop
    (single-threaded from the checker's perspective).
    """

    def __init__(self):
        self.snapshot: dict[str, str] | None = None

    def store(self, checksums: dict[str, str]) -> None:
        self.snapshot = checksums

    def compare(self, current: dict[str, str]) -> tuple[bool, list[str]]:
        """Return (all_match, list_of_mismatched_names)."""
        if self.snapshot is None:
            raise RuntimeError("No snapshot taken yet; call action='snapshot' first")
        mismatches = [
            name for name, digest in current.items()
            if self.snapshot.get(name) != digest
        ]
        # Also flag names present in snapshot but missing in current
        missing = [n for n in self.snapshot if n not in current]
        all_bad = mismatches + missing
        return len(all_bad) == 0, all_bad

    def reset(self) -> None:
        self.snapshot = None


@router.post("/weight_checker")
async def weight_checker(raw_request: Request) -> JSONResponse:
    """Snapshot, compare, or checksum model weights via SHA-256.

    Request body::

        {"action": "snapshot"}   -> take a fresh weight digest and store it
        {"action": "compare"}    -> compare current weights against stored snapshot
        {"action": "checksum"}   -> return per-tensor SHA-256 without storing
        {"action": "reset"}      -> clear the stored snapshot

    Responses (all 200 on success):

    * **snapshot**: ``{"status": "snapshotted", "n_tensors": int}``
    * **compare**:  ``{"match": bool, "mismatches": [str]}``
    * **checksum**: ``{"checksums": {name: hex_str}}``
    * **reset**:    ``{"status": "reset"}``

    Use case in RL: call ``snapshot`` right before a weight update, then
    ``compare`` after ``finish_weight_update`` to verify the engine's weights
    actually changed (non-zero NCCL transfer).
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    action = body.get("action")
    if action not in ("snapshot", "compare", "checksum", "reset"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"action must be one of snapshot|compare|checksum|reset, got {action!r}",
        )

    client = engine_client(raw_request)
    checker: _WeightCheckerState = raw_request.app.state.weight_checker

    if action == "reset":
        checker.reset()
        return JSONResponse(content={"status": "reset"})

    # All other actions require computing current checksums.
    checksums: dict = await client.compute_weight_checksums()

    if action == "snapshot":
        checker.store(checksums)
        return JSONResponse(content={"status": "snapshotted", "n_tensors": len(checksums)})

    if action == "checksum":
        return JSONResponse(content={"checksums": checksums})

    # action == "compare"
    try:
        match, mismatches = checker.compare(checksums)
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc

    return JSONResponse(content={"match": match, "mismatches": mismatches})


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


@router.get("/weight_update_active")
async def weight_update_active_endpoint(raw_request: Request) -> JSONResponse:
    """Return whether a weight update is currently in progress.

    Response::

        {"weight_update_active": bool}

    Use this to poll for completion or detect hung updates.
    """
    sm: RLStateMachineState = raw_request.app.state.rl_state
    return JSONResponse(content=sm.to_dict())


def attach_router(app: FastAPI):
    # Initialize per-request state objects on the app.
    if not hasattr(app.state, "weight_version"):
        app.state.weight_version = _WeightVersionState()
    if not hasattr(app.state, "weight_checker"):
        app.state.weight_checker = _WeightCheckerState()
    if not hasattr(app.state, "rl_state"):
        app.state.rl_state = RLStateMachineState()
    # Initialize Prometheus RL metrics with default label values so they appear
    # in /metrics from server startup (labeled metrics are not collected until
    # their first label-set is created).
    rl_weight_gen.labels(engine=_ENGINE_IDX).set(0)
    rl_weight_update_active.labels(engine=_ENGINE_IDX).set(0)
    rl_weight_update_total.labels(engine=_ENGINE_IDX)   # creates child at 0
    rl_weight_update_duration_seconds.labels(engine=_ENGINE_IDX)  # creates child
    app.include_router(router)
