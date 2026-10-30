# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)

_STATUS_TIMEOUT_SECONDS = 5.0


router = APIRouter()


def engine_client(request: Request) -> EngineClient | None:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    snapshot_client = getattr(raw_request.app.state, "engine_snapshot_client", None)
    if snapshot_client is not None:
        try:
            status = await asyncio.to_thread(
                snapshot_client.request,
                "status",
                timeout=_STATUS_TIMEOUT_SECONDS,
            )
        except Exception:
            return Response(status_code=503)
        if status["state"] != "READY":
            # The API server is alive in every non-READY snapshot state;
            # /snapshot/ready is the endpoint that gates serving.
            return Response(status_code=200)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)
