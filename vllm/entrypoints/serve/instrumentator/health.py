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
    """Readiness probe. Returns 503 during shutdown/drain so that
    load balancers stop sending new traffic."""
    if getattr(raw_request.app.state, "draining", False):
        return Response(status_code=503)
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


@router.get("/live", response_class=Response)
async def live(raw_request: Request) -> Response:
    """Liveness probe. Returns 200 as long as the process is alive,
    even during graceful shutdown/drain. Only returns 503 when the
    engine has encountered a fatal error."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always alive.
        return Response(status_code=200)
    if client.is_engine_dead:
        return Response(status_code=503)
    return Response(status_code=200)
