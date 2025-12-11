# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.dev.protocol import ReconfigureRequest
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    # get POST params
    level = raw_request.query_params.get("level", "1")
    await engine_client(raw_request).sleep(int(level))
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
    logger.info("check whether the engine is sleeping")
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


@router.post("/reconfigure")
async def reconfigure(request: ReconfigureRequest, raw_request: Request):
    success = await engine_client(raw_request).reconfigure_async(
        max_num_seqs=request.max_num_seqs,
        max_num_batched_tokens=request.max_num_batched_tokens,
    )
    return JSONResponse(content={"success": success})


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    logger.warning(
        "SECURITY WARNING: Development endpoints are enabled! "
        "This should NOT be used in production!"
    )

    app.include_router(router)
