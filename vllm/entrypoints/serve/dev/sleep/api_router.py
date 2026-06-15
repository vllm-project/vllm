# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def optional_tags(payload: dict, key: str) -> list[str] | None:
    tags = payload.get(key)
    if tags is None:
        return None
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise HTTPException(status_code=400, detail=f"{key} must be a list of strings")
    if not tags:
        raise HTTPException(status_code=400, detail=f"{key} must not be empty")
    return tags


def required_tags(payload: dict, key: str = "tags") -> list[str]:
    tags = optional_tags(payload, key)
    if tags is None:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return tags


def required_int_list(payload: dict, key: str) -> list[int]:
    value = payload.get(key)
    if value is None:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be a list of integers",
        )
    if not value:
        raise HTTPException(status_code=400, detail=f"{key} must not be empty")
    if any(item < 0 for item in value):
        raise HTTPException(status_code=400, detail=f"{key} must be non-negative")
    if len(set(value)) != len(value):
        raise HTTPException(
            status_code=400, detail=f"{key} must not contain duplicates"
        )
    return value


def optional_level(payload: dict, default: int = 1) -> int:
    value = payload.get("level", default)
    if value not in (1, 2):
        raise HTTPException(status_code=400, detail="level must be 1 or 2")
    return value


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    # get POST params
    level = raw_request.query_params.get("level", "1")
    mode = raw_request.query_params.get("mode", "abort")
    await engine_client(raw_request).sleep(int(level), mode)
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


@router.post("/sleep_ep_ranks_tags")
async def sleep_ep_ranks_by_tags(raw_request: Request):
    payload = await raw_request.json()
    sleeping_ep_ranks = required_int_list(payload, "sleeping_ep_ranks")
    tags = required_tags(payload, "tags")
    level = optional_level(payload)

    await engine_client(raw_request).collective_rpc(
        "sleep_ep_ranks_by_tags",
        kwargs={
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
            "level": level,
        },
    )
    return JSONResponse(
        content={
            "ok": True,
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
            "level": level,
        }
    )


@router.post("/wake_up_ep_ranks_tags")
async def wake_up_ep_ranks_by_tags(raw_request: Request):
    payload = await raw_request.json()
    sleeping_ep_ranks = required_int_list(payload, "sleeping_ep_ranks")
    tags = required_tags(payload, "tags")
    level = optional_level(payload)

    await engine_client(raw_request).collective_rpc(
        "wake_up_ep_ranks",
        kwargs={
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
            "level": level,
        },
    )
    return JSONResponse(
        content={
            "ok": True,
            "sleeping_ep_ranks": sleeping_ep_ranks,
            "tags": tags,
            "level": level,
        }
    )


def attach_router(app: FastAPI):
    app.include_router(router)
