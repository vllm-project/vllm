# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import uuid
from http import HTTPStatus
from typing import TypeAlias

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()

ParamType: TypeAlias = type | tuple[type, ...]
INSTRUCTION_PARAMS: dict[str, dict[str, dict[str, ParamType]]] = {
    "pause": {
        "required": {"timeout": (int, float)},
        "optional": {"exclude_engine_index": list},
    }
}


def _validate_instruction_params(instruction: str, params: dict) -> None:
    def _format_expected_type(expected_type: type | tuple[type, ...]) -> str:
        if isinstance(expected_type, tuple):
            return " | ".join(t.__name__ for t in expected_type)
        return expected_type.__name__

    rule = INSTRUCTION_PARAMS[instruction]
    for key, expected_type in rule["required"].items():
        expected_name = _format_expected_type(expected_type)
        if key not in params or not isinstance(params[key], expected_type):
            raise HTTPException(
                400,
                detail=f"Missing or invalid {key} value. Expected: {expected_name}.",
            )
    for key, expected_type in rule["optional"].items():
        expected_name = _format_expected_type(expected_type)
        if key in params and not isinstance(params[key], expected_type):
            raise HTTPException(
                400,
                detail=f"Invalid {key} value. Expected: {expected_name}.",
            )


def _validate_fault_tolerance_payload(body: dict):
    instruction = body.get("instruction")
    params = body.get("params")

    if instruction is None or params is None:
        raise HTTPException(400, detail="'instruction' and 'params' are required.")

    if instruction not in INSTRUCTION_PARAMS:
        raise HTTPException(400, detail="Invalid 'instruction'.")

    _validate_instruction_params(instruction, params)
    return instruction, params


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

    instruction, params = _validate_fault_tolerance_payload(body)
    ft_request = FaultToleranceRequest(str(uuid.uuid4()), instruction, params)

    client = engine_client(raw_request)
    try:
        ft_result = await client.handle_fault(ft_request)
    except Exception as e:
        logger.error("Failed to handle fault: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Failed to handle fault.",
        ) from e

    if ft_result.success:
        return JSONResponse({"message": "Instruction executed successfully."})

    raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        detail=f"Instruction execution failed. Reason: {ft_result.reason}",
    )


@router.get("/fault_tolerance/status")
async def get_fault_info(
    raw_request: Request,
):
    client = engine_client(raw_request)
    engine_status_dict = await client.get_fault_info()
    return JSONResponse(content=engine_status_dict)


def register_fault_tolerance_api_router(app: FastAPI):
    app.include_router(router)
