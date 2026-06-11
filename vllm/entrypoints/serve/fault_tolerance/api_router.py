# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import uuid
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

logger = init_logger(__name__)

router = APIRouter()

_ALLOWED_INSTRUCTIONS = {"retry"}


def _validate_payload(body: dict) -> tuple[str, dict]:
    instruction = body.get("instruction")
    params = body.get("params")
    if not instruction or not isinstance(params, dict):
        raise HTTPException(400, "'instruction' and 'params' are required.")
    if instruction not in _ALLOWED_INSTRUCTIONS:
        raise HTTPException(400, f"Invalid instruction: '{instruction}'.")
    if "timeout" not in params or not isinstance(params["timeout"], (int, float)):
        raise HTTPException(400, "Missing or invalid 'timeout' parameter.")
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
        raise HTTPException(400, "Invalid JSON format") from e

    instruction, params = _validate_payload(body)
    ft_request = FaultToleranceRequest(
        instruction=instruction,
        params=params,
        request_id=str(uuid.uuid4()),
    )

    client: EngineClient = raw_request.app.state.engine_client
    try:
        ft_result = await client.handle_fault(ft_request)
    except Exception as e:
        logger.error("Failed to handle fault: %s", e)
        raise HTTPException(500, "Failed to handle fault.") from e

    if ft_result.success:
        return JSONResponse({"message": "Instruction executed successfully."})
    raise HTTPException(500, f"Instruction failed: {ft_result.reason}")


@router.get("/fault_tolerance/status")
async def get_status(raw_request: Request):
    client: EngineClient = raw_request.app.state.engine_client
    return JSONResponse(content=await client.get_status())


def register_fault_tolerance_api_router(app: FastAPI):
    app.include_router(router)
