# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from fastapi import APIRouter, Request
from fastapi.responses import Response

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
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


@router.get("/liveness", response_class=Response)
async def liveness(raw_request: Request) -> Response:
    """Check that every worker on the managed engines can answer an RPC."""
    client = engine_client(raw_request)
    if client is None:
        return Response(status_code=200)
    timeout = envs.VLLM_WORKER_LIVENESS_TIMEOUT_S
    if not math.isfinite(timeout) or timeout <= 0:
        logger.error("VLLM_WORKER_LIVENESS_TIMEOUT_S must be finite and positive.")
        return Response(status_code=503)
    try:
        await client.check_worker_liveness(timeout)
    except Exception:
        logger.debug("Worker liveness probe failed.", exc_info=True)
        return Response(status_code=503)
    return Response(status_code=200)
