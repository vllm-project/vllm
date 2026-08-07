# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
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

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e  # noqa: B904

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
    await engine_client(raw_request).start_weight_update()
    return JSONResponse(content={"message": "Weight update started"})


@router.post("/start_draft_weight_update")
async def start_draft_weight_update(raw_request: Request):
    await engine_client(raw_request).start_draft_weight_update()
    return JSONResponse(content={"message": "Draft weight update started"})


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
async def finish_weight_update(
    raw_request: Request,
    weight_version: Annotated[str | None, Body(embed=True)] = None,
):
    await engine_client(raw_request).finish_weight_update(weight_version)
    return JSONResponse(content={"message": "Weight update finished"})


@router.post("/update_weight_version")
async def update_weight_version(
    raw_request: Request,
    new_version: Annotated[str, Body(embed=True)],
):
    await engine_client(raw_request).update_weight_version(new_version)
    return JSONResponse(content={"success": True, "new_version": new_version})


@router.get("/weight_info")
async def weight_info(raw_request: Request):
    weight_version = await engine_client(raw_request).get_weight_version()
    return JSONResponse(content={"weight_version": weight_version})


# ---------------------------------------------------------------------------
# Weight checksum (checksum / reset / compare)
# ---------------------------------------------------------------------------


def _merge_weight_checksums(
    per_engine: list[dict[str, str]],
) -> dict[str, str]:
    """Merge engine results using the complete parallel-rank-qualified key."""
    merged: dict[str, str] = {}
    for engine_checksums in per_engine:
        duplicate_keys = merged.keys() & engine_checksums.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys))
            raise RuntimeError(f"Duplicate weight checksum keys: {duplicates}")
        merged.update(engine_checksums)
    return merged


class _WeightCheckerState:
    """Store the first checksum result in a verification cycle.

    Operations that mutate the baseline must be externally serialized.
    """

    def __init__(self):
        self.baseline: dict[str, str] | None = None

    def store_if_absent(self, checksums: dict[str, str]) -> bool:
        """Store checksums unless a comparison baseline already exists."""
        if self.baseline is not None:
            return False
        self.baseline = dict(checksums)
        return True

    def has_baseline(self) -> bool:
        """Return whether a comparison baseline is currently stored."""
        return self.baseline is not None

    def compare(self, current: dict[str, str]) -> tuple[bool, list[str]]:
        """Compare the current checksums with the stored baseline.

        Args:
            current: Complete rank-qualified keys mapped to SHA-256 digests.

        Returns:
            A tuple containing whether all tensors match and the names of changed,
            added, or missing tensors.

        Raises:
            RuntimeError: If no baseline has been stored.
        """
        if self.baseline is None:
            raise RuntimeError("No checksum baseline; call action='checksum' first")
        mismatches = sorted(
            key
            for key in self.baseline.keys() | current.keys()
            if self.baseline.get(key) != current.get(key)
        )
        # Compare is one-shot: clear the baseline so a second compare fails.
        self.baseline = None
        return not mismatches, mismatches


@router.post("/weight_checker")
async def weight_checker(raw_request: Request) -> JSONResponse:
    """Checksum, reset, or compare model weights.

    Request body::

        {"action": "compare"}    -> compare current weights against the baseline
        {"action": "checksum"}   -> return SHA-256 and store the first baseline
        {"action": "reset"}      -> overwrite GPU weights with random values

    Responses (all 200 on success):

    * **compare**:  ``{"match": bool, "mismatches": [str]}``
    * **checksum**: ``{"checksums": {name: hex_str}}``
    * **reset**:    ``{"status": "reset"}``

    Use case in RL: checksum the current weights, reset them, transfer the
    original weights, checksum again, and compare with the first checksum.
    A successful transfer is expected to match the baseline.
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    action = body.get("action")
    if action not in ("compare", "checksum", "reset"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"action must be one of checksum|reset|compare, got {action!r}",
        )

    client = engine_client(raw_request)
    checker: _WeightCheckerState = raw_request.app.state.weight_checker

    if action == "reset":
        # Overwrite every weight-bearing tensor with random values on the GPU
        await client.reset_weights()
        return JSONResponse(content={"status": "reset"})

    # Avoid an expensive checksum RPC when compare cannot succeed. compare()
    # repeats this check to keep the state object safe when used directly.
    if action == "compare" and not checker.has_baseline():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="No checksum baseline; call action='checksum' first",
        )

    per_engine: list[dict[str, str]] = (
        await client.compute_weight_checksums_all()
    )
    if not per_engine:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No engine returned weight checksums",
        )
    try:
        checksums = _merge_weight_checksums(per_engine)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(exc),
        ) from exc

    if action == "checksum":
        baseline_created = checker.store_if_absent(checksums)
        return JSONResponse(
            content={
                "checksums": checksums,
                "engines": per_engine,
                "baseline_created": baseline_created,
            }
        )

    # action == "compare"
    try:
        match, mismatches = checker.compare(checksums)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(exc),
        ) from exc

    return JSONResponse(
        content={
            "match": match,
            "mismatches": mismatches,
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
    # Initialize per-request state objects on the app.
    if not hasattr(app.state, "weight_checker"):
        app.state.weight_checker = _WeightCheckerState()

    app.include_router(router)
