# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine import PauseMode

from .metrics import sleep_mode_operation_metrics

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/sleep")
async def sleep(
    raw_request: Request,
    level: Annotated[int, Query(ge=0, le=2)] = 1,
    mode: Annotated[PauseMode, Query()] = "abort",
) -> JSONResponse:
    with sleep_mode_operation_metrics().record("sleep"):
        await engine_client(raw_request).sleep(level, mode)
    return JSONResponse(content={"status": "sleeping", "level": level})


@router.post("/release_kv_cache_memory")
async def release_kv_cache_memory(raw_request: Request) -> JSONResponse:
    with sleep_mode_operation_metrics().record("release_kv_cache_memory"):
        await engine_client(raw_request).release_kv_cache_memory()
    return JSONResponse(content={"status": "kv_cache_released"})


@router.post("/wake_up")
async def wake_up(raw_request: Request) -> JSONResponse:
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    with sleep_mode_operation_metrics().record("wake"):
        fully_awake = await engine_client(raw_request).wake_up(tags)
    return JSONResponse(
        content={"status": "awake" if fully_awake else "sleeping", "tags_woken": tags}
    )


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


def attach_router(app: FastAPI):
    app.include_router(router)
