# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/kv_event_sources")
async def get_kv_event_sources(raw_request: Request):
    """Report the resolved KV-event publisher endpoint per DP rank.

    Endpoints are assigned at bind time by each EngineCore (ephemeral
    when configured with port 0) and published to the frontend in the
    engine ready response, so this route answers from a static table
    with no engine round-trip.
    """
    sources = engine_client(raw_request).get_kv_event_sources()
    return JSONResponse(content={"sources": sources})


def attach_router(app: FastAPI):
    app.include_router(router)
