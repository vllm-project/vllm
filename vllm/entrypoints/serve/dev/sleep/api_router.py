# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    """Put the engine to sleep, releasing GPU memory.

    Returns a JSON body describing the transition. The ``await`` below
    completes only once the engine core has applied the sleep (the utility
    call resolves on the engine-core response), so the engine is actually
    asleep by the time this returns.
    """
    level = int(raw_request.query_params.get("level", "1"))
    mode = raw_request.query_params.get("mode", "abort")
    client = engine_client(raw_request)

    t0 = time.perf_counter()
    await client.sleep(level, mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return JSONResponse(
        content={
            "status": "sleeping",
            "level": level,
            "elapsed_ms": round(elapsed_ms, 2),
        }
    )


@router.post("/wake_up")
async def wake_up(raw_request: Request):
    """Wake the engine, re-mapping GPU memory for the given tags.

    With no tags, all tags are woken. With a subset of tags (e.g. only
    ``weights``), the engine remains sleeping until the remaining tags are
    woken; the response ``status`` reflects this (``sleeping`` vs ``awake``).
    """
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    client = engine_client(raw_request)

    t0 = time.perf_counter()
    await client.wake_up(tags)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # A partial wake (subset of tags) leaves the engine still sleeping; report
    # the resulting state so orchestrators can sequence staged wakes correctly.
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
