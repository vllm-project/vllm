# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import validate_json_request
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post(
    "/fault_tolerance/apply",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def process_fault_tolerance_instruction(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e

    client = engine_client(raw_request)

    fault_tolerance_instruction = body.get("fault_tolerance_instruction")
    fault_tolerance_timeout = body.get("fault_tolerance_timeout")
    dynamic_fault_tolerance_params = body.get("fault_tolerance_params", {})

    if fault_tolerance_instruction is None or fault_tolerance_timeout is None:
        raise HTTPException(
            status_code=400,
            detail="Both 'fault_tolerance_instruction' and "
            "'fault_tolerance_timeout' are required.",
        )

    if not isinstance(fault_tolerance_instruction, str):
        raise HTTPException(
            status_code=400, detail="'fault_tolerance_instruction' must be a string."
        )
    # Supported instructions: ["pause", "retry"].
    # More instruction types may be added in future updates.
    elif fault_tolerance_instruction not in ["pause", "retry"]:
        raise HTTPException(
            status_code=400, detail="Invalid 'fault_tolerance_instruction' value."
        )

    if not isinstance(fault_tolerance_timeout, int) or fault_tolerance_timeout <= 0:
        raise HTTPException(
            status_code=400,
            detail="'fault_tolerance_timeout' must be a positive integer.",
        )
    try:
        success = await client.handle_fault(
            fault_tolerance_instruction,
            fault_tolerance_timeout,
            **dynamic_fault_tolerance_params,
        )
        if success:
            return JSONResponse(
                {
                    "message": "Instruction executed successfully.",
                }
            )
        else:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="Instruction execution failed.",
            )

    except Exception as e:
        logger.error("Failed to handle fault: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Failed to handle fault.",
        ) from e


@router.get("/fault_tolerance/status")
async def get_fault_info(
    raw_request: Request,
):
    client = engine_client(raw_request)
    engine_status_dict = await client.get_fault_info()
    return JSONResponse(content=engine_status_dict)


def attach_router(app: FastAPI):
    app.include_router(router)
