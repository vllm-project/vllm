# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.plugins.endpoint_plugins.interface import endpoint_plugins_ready
from vllm.v1.engine.exceptions import EngineDeadError

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
    """Process-local engine and plugin readiness, excluding draining servers."""
    state = raw_request.app.state
    if getattr(state, "draining", False):
        return Response(status_code=503)
    client = engine_client(raw_request)
    if client is not None:
        try:
            await client.check_health()
        except EngineDeadError:
            return Response(status_code=503)
    if getattr(state, "draining", False) or not endpoint_plugins_ready(state):
        return Response(status_code=503)
    return Response(status_code=200)
