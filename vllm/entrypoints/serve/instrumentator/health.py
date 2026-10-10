# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import (
    EngineDeadError,
    EngineSleepingError,
    EngineUnhealthyError,
)

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


@router.get("/ready", response_class=Response)
async def ready(raw_request: Request) -> Response:
    """GPU execution and EngineCore progress readiness check."""
    client = engine_client(raw_request)
    if client is None:
        return Response(status_code=200)
    try:
        await client.check_health_gpu()
        return Response(status_code=200)
    except EngineSleepingError:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "sleeping"},
        )
    except (EngineDeadError, EngineUnhealthyError):
        return Response(status_code=503)
