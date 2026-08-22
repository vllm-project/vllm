# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.serve.elastic_ep.middleware import get_scaling_elastic_ep
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post(
    "/scale_elastic_ep",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def scale_elastic_ep(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e

    new_data_parallel_size = body.get("new_data_parallel_size")
    drain_timeout = body.get("drain_timeout", 120)  # Default 2 minutes

    if new_data_parallel_size is None:
        raise HTTPException(
            status_code=400, detail="new_data_parallel_size is required"
        )

    if not isinstance(new_data_parallel_size, int) or new_data_parallel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="new_data_parallel_size must be a positive integer",
        )

    if not isinstance(drain_timeout, int) or drain_timeout <= 0:
        raise HTTPException(
            status_code=400, detail="drain_timeout must be a positive integer"
        )

    client = engine_client(raw_request)
    try:
        await client.scale_elastic_ep(new_data_parallel_size, drain_timeout)
        return JSONResponse(
            {
                "message": f"Scaled to {new_data_parallel_size} data parallel engines",
            }
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail="Scale failed due to request drain timeout "
            f"after {drain_timeout} seconds",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Scale failed: %s", e)
        raise HTTPException(status_code=500, detail="Scale failed") from e


@router.post("/is_scaling_elastic_ep")
async def is_scaling_elastic_ep(raw_request: Request):
    # External operation status comes from the shared store. Middleware gating
    # remains process-local, so scaling requests must reach every old-rank API.
    try:
        phase = await engine_client(raw_request).get_external_elastic_ep_phase()
    except Exception as e:
        logger.warning("Failed to query external Elastic EP phase: %s", e)
        raise HTTPException(
            status_code=503,
            detail="External Elastic EP status is temporarily unavailable",
        ) from e
    if phase is None:
        # Non-external EEP modes retain the process-local middleware state.
        is_scaling = get_scaling_elastic_ep()
        phase = "committing" if is_scaling else "idle"
    else:
        is_scaling = phase in ("preparing", "committing")
    return JSONResponse({"is_scaling_elastic_ep": is_scaling, "phase": phase})


def attach_router(app: FastAPI):
    app.include_router(router)
