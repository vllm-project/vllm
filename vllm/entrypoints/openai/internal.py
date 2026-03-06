# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.post("/internal/kv_connector_refresh_lease")
async def kv_connector_refresh_lease(raw_request: Request) -> Response:
    """Receive KV lease refresh requests from D workers.

    D workers POST here periodically while requests are queued, before the
    KV transfer begins, to prevent P from expiring and freeing KV blocks
    prematurely.
    """
    try:
        body = await raw_request.json()
        request_id: str = body.get("request_id")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "kv_connector_refresh_lease: failed to parse request body: %s", e
        )
        return Response(status_code=400)

    engine_client = raw_request.app.state.engine_client
    await engine_client.call_utility_async("kv_connector_refresh_lease", request_id)
    return Response(status_code=200)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
