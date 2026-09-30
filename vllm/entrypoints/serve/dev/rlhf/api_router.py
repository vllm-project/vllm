# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdatePayload,
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
        raw_request: The incoming FastAPI request, used to reach the engine
            client on the app state.
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
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object"
        )

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
async def init_weight_transfer_engine(
    raw_request: Request,
    init_info: Annotated[dict[str, Any], Body(embed=True)],
):
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
async def update_weights(
    raw_request: Request,
    update_info: Annotated[WeightTransferUpdatePayload, Body(embed=True)],
):
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


@router.post("/weight_checker")
async def weight_checker(
    raw_request: Request,
    action: Annotated[Literal["checksum", "reset", "compare"], Body(embed=True)],
    baseline: Annotated[dict[str, str] | None, Body(embed=True)] = None,
) -> JSONResponse:
    """Checksum, reset, or compare model weights against a baseline."""
    client = engine_client(raw_request)
    if action == "reset":
        await client.collective_rpc("reset_weights")
        return JSONResponse(content={"status": "reset"})
    if action == "compare" and baseline is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="action='compare' requires a 'baseline' object",
        )

    checksums: dict[str, str] = {}
    for worker_checksums in await client.compute_weight_checksums():
        if duplicates := checksums.keys() & worker_checksums.keys():
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail=f"Duplicate weight checksum keys: {sorted(duplicates)}",
            )
        checksums.update(worker_checksums)
    if action == "checksum":
        return JSONResponse(content={"checksums": checksums})

    assert baseline is not None
    mismatches = sorted(
        key
        for key in checksums.keys() | baseline.keys()
        if checksums.get(key) != baseline.get(key)
    )
    return JSONResponse(content={"match": not mismatches, "mismatches": mismatches})


@router.get("/get_world_size")
async def get_world_size(
    raw_request: Request,
    include_dp: bool = Query(True),
):
    """Get the world size from the parallel config.

    Args:
        raw_request: The incoming FastAPI request, used to reach the engine
            client on the app state.
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
    app.include_router(router)
