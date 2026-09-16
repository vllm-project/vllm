# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import ALLOWED_FT_INSTRUCTIONS, FaultToleranceRequest

logger = init_logger(__name__)

router = APIRouter()


def _validate_payload(body: dict) -> tuple[str, dict, str]:
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    instruction = body.get("instruction")
    if not instruction:
        raise HTTPException(400, "'instruction' is required.")
    if instruction not in ALLOWED_FT_INSTRUCTIONS:
        raise HTTPException(400, f"Invalid instruction: '{instruction}'.")
    params = body.get("params", {})
    if not isinstance(params, dict):
        raise HTTPException(400, "'params' must be an object.")

    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise HTTPException(400, "'request_id' must be a non-empty string.")
    return instruction, params, request_id


@router.post(
    "/v1/fault_tolerance/apply",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.ACCEPTED.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
    },
)
async def process_fault_tolerance_instruction(
    raw_request: Request, background_tasks: BackgroundTasks
):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(400, "Invalid JSON format") from e

    instruction, params, request_id = _validate_payload(body)
    # One recovery round shares one request_id, which namespaces that round's
    # coordination keys. The orchestrator must send the same request_id to
    # every engine in a round; the engines verify this during recovery.
    ft_request = FaultToleranceRequest(
        instruction=instruction,
        params=params,
        request_id=request_id,
    )

    client: EngineClient = raw_request.app.state.engine_client
    # Recovery runs cross-rank collective ops that only complete once every rank
    # has been dispatched. Run it in the background and return immediately so the
    # orchestrator can dispatch to all ranks without blocking; completion is
    # observed by polling GET /v1/fault_tolerance/status.
    background_tasks.add_task(_run_fault_recovery, client, ft_request)
    return JSONResponse(
        status_code=HTTPStatus.ACCEPTED.value,
        content={
            "message": "Request accepted; poll /v1/fault_tolerance/status for updates.",
            "request_id": ft_request.request_id,
        },
        background=background_tasks,
    )


async def _run_fault_recovery(
    client: EngineClient, ft_request: FaultToleranceRequest
) -> None:
    """Drive recovery to completion after the 202 response is sent."""
    try:
        result = await client.handle_fault(ft_request)
    except Exception:
        logger.exception("[FT] Recovery dispatch failed.")
        return
    if not result.success:
        logger.error(
            "[FT] Recovery failed for request %s: %s",
            ft_request.request_id,
            result.reason,
        )


@router.get("/v1/fault_tolerance/status")
async def get_status(raw_request: Request):
    client: EngineClient = raw_request.app.state.engine_client
    return JSONResponse(content=await client.get_status())


def register_fault_tolerance_api_router(app: FastAPI):
    app.include_router(router)
