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
    """Return 503 if the engine is dead, the server is draining or an endpoint
    plugin is not ready."""
    client = engine_client(raw_request)
    # Render-only servers have no engine to check.
    if client is not None:
        try:
            await client.check_health()
        except EngineDeadError:
            return Response(status_code=503)
    state = raw_request.app.state
    if getattr(state, "draining", False) or not endpoint_plugins_ready(state):
        return Response(status_code=503)
    return Response(status_code=200)
