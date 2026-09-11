# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/drop_peer")
async def drop_peer(raw_request: Request):
    """Release all resources tied to a remote engine declared dead.

    Intended for disaggregated (P/D) deployments: the router calls this on
    the *peer* of an engine that went down (e.g. P's `/drop_peer` with the
    dead decode engine's id, or vice versa). The whole engine's requests
    routed to the dead peer are aborted and their blocks freed, and the
    connector's per-peer NIXL state is released so a replacement reusing
    the same engine id can re-handshake.

    Requests are attributed to the peer via
    `kv_transfer_params["decode_engine_id"]` (P side, router-provided);
    untagged requests are left untouched.
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e

    engine_id = body.get("engine_id")
    if not isinstance(engine_id, str) or not engine_id:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing or invalid 'engine_id' in request body",
        )

    await engine_client(raw_request).drop_peer(engine_id)
    return JSONResponse(content={"status": "ok"})


def attach_router(app: FastAPI):
    app.include_router(router)
