# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
from http import HTTPStatus

import msgspec
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleConflict,
    ExternalElasticEPScaleStatus,
)
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.elastic_ep.middleware import (
    get_elastic_ep_rank_retired,
    get_scaling_elastic_ep,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


def _optional_identifier(body: dict, name: str) -> str | None:
    value = body.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be a non-empty string of at most 256 characters",
        )
    return value


def _status_payload(status: ExternalElasticEPScaleStatus) -> dict:
    payload = msgspec.to_builtins(status)
    payload["is_scaling_elastic_ep"] = status.active
    return payload


@router.post(
    "/scale_elastic_ep",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.CONFLICT.value: {"model": ErrorResponse},
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
    operation_id = _optional_identifier(body, "operation_id")
    expected_instance_id = _optional_identifier(body, "expected_instance_id")
    if (
        get_scaling_elastic_ep() or get_elastic_ep_rank_retired()
    ) and operation_id is None:
        raise HTTPException(status_code=503, detail="The model is currently scaling")

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
        status = await client.scale_elastic_ep(
            new_data_parallel_size,
            drain_timeout,
            operation_id,
            expected_instance_id,
        )
        if status is None:
            return JSONResponse(
                {
                    "message": (
                        f"Scaled to {new_data_parallel_size} data parallel engines"
                    )
                }
            )
        payload = _status_payload(status)
        payload["message"] = (
            f"Elastic EP operation is {status.phase.value} for data parallel "
            f"size {new_data_parallel_size}"
        )
        return JSONResponse(payload)
    except ExternalElasticEPScaleConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
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
        status = await engine_client(raw_request).get_external_elastic_ep_status()
    except Exception as e:
        logger.warning("Failed to query external Elastic EP status: %s", e)
        raise HTTPException(
            status_code=503,
            detail="External Elastic EP status is temporarily unavailable",
        ) from e
    if status is None:
        # Non-external EEP modes retain the process-local middleware state.
        is_scaling = get_scaling_elastic_ep()
        return JSONResponse(
            {
                "is_scaling_elastic_ep": is_scaling,
                "phase": "committing" if is_scaling else "idle",
                "epoch": None,
                "error": None,
                "requested_data_parallel_size": None,
            }
        )
    return JSONResponse(_status_payload(status))


def attach_router(app: FastAPI):
    app.include_router(router)
