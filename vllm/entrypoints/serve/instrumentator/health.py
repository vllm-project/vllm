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

        # Handle snapshot trigger and check
        if hasattr(client, "snapshot_manager") and client.snapshot_manager:
            if client.snapshot_task is None:
                # Trigger the snapshot in background
                import asyncio

                async def _run_snapshot():
                    await asyncio.to_thread(client.snapshot_manager.run_snapshot)

                client.snapshot_task = asyncio.create_task(_run_snapshot())
                logger.info("Triggered snapshot from health check.")
                return Response(status_code=503, content="Snapshot triggered")

            elif not client.snapshot_task.done():
                logger.info(
                    "Snapshot is still running, returning 503 for health check."
                )
                return Response(status_code=503, content="Snapshot in progress")

        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)
