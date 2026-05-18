# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    # get POST params
    level = raw_request.query_params.get("level", "1")
    mode = raw_request.query_params.get("mode", "abort")
    # Optional tag-wise offload: ?offload_tags=weights&offload_tags=kv_cache
    # When provided, the engine performs a selective sleep instead of
    # using the legacy `level` semantics. Blank values (e.g. the request
    # `?offload_tags=`) are filtered out — Starlette returns those as the
    # empty string and forwarding them to the engine would record `''`
    # as a real selective-sleep tag, leaving the executor stuck in
    # `is_sleeping=True`. If every value is blank we fall back to the
    # legacy level-based path (offload_tags=None) instead of silently
    # turning the request into a pure pause.
    offload_tags_raw = [
        t for t in raw_request.query_params.getlist("offload_tags") if t.strip()
    ]
    offload_tags = offload_tags_raw if offload_tags_raw else None
    await engine_client(raw_request).sleep(int(level), mode, offload_tags=offload_tags)
    # FIXME: in v0 with frontend multiprocessing, the sleep command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


@router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    await engine_client(raw_request).wake_up(tags)
    # FIXME: in v0 with frontend multiprocessing, the wake-up command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
