# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    """Return the engine client attached to the application."""
    return request.app.state.engine_client


@router.get("/kv_event_sources")
async def get_kv_event_sources(raw_request: Request):
    """Discovery info for enabled ZMQ KV-cache event publishers."""
    sources = await engine_client(raw_request).get_kv_event_sources()
    return JSONResponse(sources)


def attach_router(app: FastAPI):
    """Register KV-event discovery routes on the application."""
    app.include_router(router)
