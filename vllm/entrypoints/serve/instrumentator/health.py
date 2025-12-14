# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check.

    Returns 503 if:
    - Warmup is still in progress (warmup_complete is False)
    - The engine is dead
    """
    # Check if warmup is complete (defaults to True if not set)
    warmup_complete = getattr(raw_request.app.state, "warmup_complete", True)
    if not warmup_complete:
        return Response(status_code=503)

    try:
        await engine_client(raw_request).check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


def attach_router(app):
    app.include_router(router)
