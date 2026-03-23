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


@router.post("/suspend")
async def suspend(raw_request: Request):
    mode = raw_request.query_params.get("mode", "abort")
    if mode not in ("abort", "wait", "keep"):
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid suspend mode '{mode}'. "
                "Must be one of 'abort', 'wait', 'keep'."
            },
        )
    logger.info("Suspending engine with mode: %s", mode)
    await engine_client(raw_request).suspend(mode)
    return Response(status_code=200)


@router.post("/resume")
async def resume(raw_request: Request):
    logger.info("Resuming engine from checkpoint suspend.")
    await engine_client(raw_request).resume()
    return Response(status_code=200)


@router.get("/is_suspended")
async def is_suspended(raw_request: Request):
    is_suspended = await engine_client(raw_request).is_checkpoint_suspended()
    return JSONResponse(content={"is_suspended": is_suspended})


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
