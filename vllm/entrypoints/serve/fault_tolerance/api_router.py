# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP endpoints for the fault tolerance framework.

Three endpoints:

* ``GET /fault_tolerance/status`` — return current per-engine health.
* ``POST /fault_tolerance/apply`` — dispatch a recovery instruction.
* ``GET /fault_tolerance/health`` — quick readiness probe.

The endpoints are thin: they look up the registered supervisor / plan and
delegate. Plans live in ``vllm.v1.fault_tolerance.plans``; the wire format
of the status response is preserved-compatible with the format established
in vllm-project/vllm#34833.
"""

from __future__ import annotations

import json
import uuid
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.utils import validate_json_request
from vllm.logger import init_logger
from vllm.v1.fault_tolerance import (
    FaultStatus,
    FaultToleranceRequest,
    FaultToleranceResult,
    get_plan,
    list_plans,
)
from vllm.v1.fault_tolerance.registry import _SUPERVISORS

logger = init_logger(__name__)

router = APIRouter(prefix="/fault_tolerance")


def _engine_client(request: Request):
    return request.app.state.engine_client


def _wire_status_payload(status_dict: dict[int, FaultStatus]) -> dict:
    """Format the engine-status dict for the wire.

    Schema mirrors vllm-project/vllm#34833 so external subscribers (Dynamo,
    monitoring, etc.) work without modification:

    .. code-block:: json

        {
          "schema_version": 1,
          "total_engines": 4,
          "engines": [
            {"id": 0, "status": "healthy"},
            {"id": 1, "status": "paused"},
            ...
          ]
        }
    """
    engines = [
        {"id": idx, "status": status.name.lower()}
        for idx, status in sorted(status_dict.items())
    ]
    return {
        "schema_version": 1,
        "total_engines": len(engines),
        "engines": engines,
    }


@router.get("/status")
async def fault_tolerance_status(raw_request: Request):
    """Return the current per-engine health snapshot."""
    client = _engine_client(raw_request)
    fetch = getattr(client, "get_fault_status", None)
    if fetch is None:
        # FT not enabled or no supervisor configured.
        return JSONResponse(_wire_status_payload({}))
    try:
        status_dict = await fetch()
    except Exception as e:
        logger.exception("Failed to fetch fault status: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Failed to fetch fault status",
        ) from e
    return JSONResponse(_wire_status_payload(status_dict))


@router.post(
    "/apply",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": dict},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": dict},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": dict},
    },
)
async def fault_tolerance_apply(raw_request: Request):
    """Dispatch a recovery instruction (pause / retry / scale_down / etc.).

    Body schema::

        {
            "instruction": "pause" | "retry" | "scale_down" | "scale_up" | ...,
            "params": {...},  # optional, plan-specific
            "request_id": "...",  # optional; auto-generated if absent
        }
    """
    if not _SUPERVISORS:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="No fault supervisor is registered. Pass --enable-fault-tolerance "
            "and select a supervisor with --fault-supervisor.",
        )
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail="Invalid JSON"
        ) from e

    instruction = body.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'instruction' must be a non-empty string",
        )
    if instruction not in list_plans():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Unknown instruction '{instruction}'. Registered: {list_plans()}"),
        )
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'params' must be a JSON object if provided",
        )
    request_id = body.get("request_id") or str(uuid.uuid4())

    ft_request = FaultToleranceRequest(
        instruction=instruction, params=params, request_id=request_id
    )

    client = _engine_client(raw_request)
    apply_fn = getattr(client, "apply_fault_instruction", None)
    if apply_fn is None:
        # Default supervisor not wired in (Phase 0). Indicate that a plan is
        # registered but no execution path exists yet.
        plan = get_plan(instruction)
        return JSONResponse(
            FaultToleranceResult(
                success=False,
                reason=(
                    "No supervisor execution path is wired into the engine "
                    "client. Plan '{}' is registered but cannot be invoked "
                    "from HTTP yet.".format(type(plan).__name__)
                ),
                request_id=request_id,
            ).__dict__,
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )

    try:
        result: FaultToleranceResult = await apply_fn(ft_request)
    except Exception as e:
        logger.exception("apply_fault_instruction failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"apply failed: {e}",
        ) from e

    return JSONResponse(
        result.__dict__,
        status_code=HTTPStatus.OK.value
        if result.success
        else HTTPStatus.BAD_REQUEST.value,
    )


@router.get("/health")
async def fault_tolerance_health(raw_request: Request):
    """Lightweight readiness probe for the FT subsystem itself."""
    return JSONResponse({"ok": True, "registered_plans": list_plans()})


def attach_router(app: FastAPI):
    """Register the fault-tolerance routes with the app."""
    app.include_router(router)
