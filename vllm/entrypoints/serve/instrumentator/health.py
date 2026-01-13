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
    """Health check. Returns 503 when draining or dead.

    Designed to be used as the readiness probe in a Kubernetes deployment
    """
    try:
        client = engine_client(raw_request)
        if await client.is_paused():
            return Response(status_code=503)
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


@router.get("/live", response_class=Response)
async def live(raw_request: Request) -> Response:
    """Liveness check. Returns 200 when draining, 503 only when dead."""
    try:
        await engine_client(raw_request).check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


def attach_router(app):
    app.include_router(router)
