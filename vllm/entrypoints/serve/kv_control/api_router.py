# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP ingress for the same explicit engine control used by Python clients."""

from typing import Any, Literal

import msgspec
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.v1.kv_hints.api import ControlInvocation
from vllm.v1.kv_hints.control import ControlError

router = APIRouter()


class ControlRequest(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    method: Literal[
        "submit",
        "operation",
        "associations",
        "events",
        "acknowledge",
        "capacity",
        "residency",
        "cancel",
    ]
    invocation: ControlInvocation
    payload: dict[str, Any]


def error_response(code: str, message: str) -> JSONResponse:
    status = {
        "wrong_engine": 404,
        "missing": 404,
        "missing_association": 404,
        "stale_epoch": 409,
        "stale_revision": 409,
        "stale_operation": 409,
        "conflict": 409,
        "event_gap": 409,
        "unsupported": 501,
        "unsupported_consumer": 501,
        "capacity": 429,
        "busy": 409,
        "acknowledged": 409,
        "payload_too_large": 413,
    }.get(code, 400)
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status
    )


@router.get("/v1/kv/capabilities")
async def capabilities(raw_request: Request):
    client: EngineClient | None = raw_request.app.state.engine_client
    if client is None:
        return {"engines": []}
    try:
        return {"engines": await client.get_kv_control_capabilities()}
    except ControlError as error:
        return error_response(error.code, str(error))


@router.post("/v1/kv/control", dependencies=[Depends(validate_json_request)])
async def control(raw_request: Request):
    settings = raw_request.app.state.vllm_config.kv_control_config
    if settings is None:
        return error_response("unsupported", "Programmable KV control is disabled")
    data = bytearray()
    async for chunk in raw_request.stream():
        if len(data) + len(chunk) > settings.max_envelope_bytes + 2048:
            return error_response(
                "payload_too_large", "Control request byte limit exceeded"
            )
        data.extend(chunk)
    try:
        request = msgspec.json.decode(data, type=ControlRequest)
    except msgspec.DecodeError as error:
        return error_response("invalid_payload", str(error)[:4096])
    client: EngineClient = raw_request.app.state.engine_client
    try:
        result = await client.kv_control(
            request.method, msgspec.to_builtins(request.invocation), request.payload
        )
        return JSONResponse(result)
    except ControlError as error:
        return error_response(error.code, str(error))


def attach_router(app: FastAPI):
    app.include_router(router)
