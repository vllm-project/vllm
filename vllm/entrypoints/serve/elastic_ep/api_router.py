# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import time
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.elastic_ep.middleware import (
    get_scaling_elastic_ep,
    set_scaling_elastic_ep,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


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
    timing_ms: dict[str, float] = {}
    total_start = time.perf_counter()
    try:
        step_start = time.perf_counter()
        body = await raw_request.json()
        timing_ms["parse_request"] = elapsed_ms(step_start)
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

    # Set scaling flag to prevent new requests
    step_start = time.perf_counter()
    set_scaling_elastic_ep(True)
    timing_ms["set_scaling_flag"] = elapsed_ms(step_start)
    scaling_flag_set = True
    client = engine_client(raw_request)
    try:
        step_start = time.perf_counter()
        await client.scale_elastic_ep(new_data_parallel_size, drain_timeout)
        timing_ms["scale_elastic_ep"] = elapsed_ms(step_start)
        step_start = time.perf_counter()
        set_scaling_elastic_ep(False)
        scaling_flag_set = False
        timing_ms["clear_scaling_flag"] = elapsed_ms(step_start)
        timing_ms["total"] = elapsed_ms(total_start)
        logger.info(
            "scale_elastic_ep to %s timing_ms=%s",
            new_data_parallel_size,
            timing_ms,
        )
        return JSONResponse(
            {
                "message": f"Scaled to {new_data_parallel_size} data parallel engines",
                "timing_ms": timing_ms,
            }
        )
    except TimeoutError as e:
        if scaling_flag_set:
            step_start = time.perf_counter()
            set_scaling_elastic_ep(False)
            scaling_flag_set = False
            timing_ms["clear_scaling_flag"] = elapsed_ms(step_start)
        timing_ms["total"] = elapsed_ms(total_start)
        logger.exception("Scale timed out timing_ms=%s", timing_ms)
        raise HTTPException(
            status_code=408,
            detail="Scale failed due to request drain timeout "
            f"after {drain_timeout} seconds",
        ) from e
    except Exception as e:
        if scaling_flag_set:
            step_start = time.perf_counter()
            set_scaling_elastic_ep(False)
            scaling_flag_set = False
            timing_ms["clear_scaling_flag"] = elapsed_ms(step_start)
        timing_ms["total"] = elapsed_ms(total_start)
        logger.exception("Scale failed timing_ms=%s", timing_ms)
        raise HTTPException(status_code=500, detail="Scale failed") from e
    finally:
        if scaling_flag_set:
            set_scaling_elastic_ep(False)


@router.post("/is_scaling_elastic_ep")
async def is_scaling_elastic_ep(raw_request: Request):
    return JSONResponse({"is_scaling_elastic_ep": get_scaling_elastic_ep()})


def attach_router(app: FastAPI):
    app.include_router(router)
