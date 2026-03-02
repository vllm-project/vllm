# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.post("/internal/nixl/lease_refresh")
async def nixl_lease_refresh(raw_request: Request) -> Response:
    """Receive KV lease refresh requests from D workers.

    D workers POST here periodically while requests are queued, before the
    NIXL transfer begins, to prevent P from expiring and freeing KV blocks
    prematurely.
    """
    try:
        body = await raw_request.json()
        request_ids: list[str] = body.get("request_ids", [])
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("nixl_lease_refresh: failed to parse request body: %s", e)
        return Response(status_code=400)

    engine_client = raw_request.app.state.engine_client
    await engine_client.call_utility_async("nixl_lease_refresh", request_ids)
    return Response(status_code=200)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
