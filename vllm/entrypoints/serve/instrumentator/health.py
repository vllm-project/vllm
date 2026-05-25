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
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()

        # Check if snapshot is enabled and in progress/pending
        if getattr(client, "snapshot_manager", None) is not None:
            snapshot_task = getattr(client, "snapshot_task", None)
            if snapshot_task is None or not snapshot_task.done():
                logger.info(
                    "Snapshot is still running or pending, returning 503 "
                    "for health check."
                )
                return Response(status_code=503, content="Snapshot in progress")

        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)
