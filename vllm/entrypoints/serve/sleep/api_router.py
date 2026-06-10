# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi import HTTPException, status

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

@router.post("/suspend")
async def suspend(raw_request: Request):
    model_save_path = raw_request.query_params.get("model_save_path")
    # 校验参数是否存在
    if model_save_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameter: model_save_path"
        )
    await engine_client(raw_request).suspend(model_save_path=model_save_path)
    return Response(status_code=200)

@router.post("/resume")
async def resume(raw_request: Request):
    # get POST params
    data_parallel_master_ip = raw_request.query_params.get("data_parallel_master_ip")
    model_path = raw_request.query_params.get("model_path")
    # 校验参数是否存在
    if data_parallel_master_ip is None or model_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required parameter: data_parallel_master_ip and model_path"
        )

    await engine_client(raw_request).resume(data_parallel_master_ip=data_parallel_master_ip, model_path=model_path)
    return Response(status_code=200)


@router.post("/device_unlock")
async def device_unlock(raw_request: Request):
    await engine_client(raw_request).device_unlock()
    return Response(status_code=200)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
