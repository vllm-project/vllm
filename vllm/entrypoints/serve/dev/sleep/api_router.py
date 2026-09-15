# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Annotated

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine import PauseMode

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
    start = time.perf_counter()
    await engine_client(raw_request).sleep(level, mode)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return JSONResponse(
        content={
            "status": "sleeping",
            "level": level,
            "elapsed_ms": round(elapsed_ms, 2),
        }
    )


@router.post("/wake_up")
async def wake_up(raw_request: Request) -> JSONResponse:
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    client = engine_client(raw_request)
    start = time.perf_counter()
    await client.wake_up(tags)
    elapsed_ms = (time.perf_counter() - start) * 1000
    still_sleeping = await client.is_sleeping()
    return JSONResponse(
        content={
            "status": "sleeping" if still_sleeping else "awake",
            "tags_woken": tags,
            "elapsed_ms": round(elapsed_ms, 2),
        }
    )


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


def attach_router(app: FastAPI):
    app.include_router(router)
