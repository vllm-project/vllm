# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/start_profile")
async def start_profile(raw_request: Request):
    logger.info("Starting profiler...")
    await engine_client(raw_request).start_profile()
    logger.info("Profiler started.")
    return Response(status_code=200)


@router.post("/stop_profile")
async def stop_profile(raw_request: Request):
    logger.info("Stopping profiler...")
    await engine_client(raw_request).stop_profile()
    logger.info("Profiler stopped.")
    return Response(status_code=200)


def attach_router(app: FastAPI):
    if envs.VLLM_TORCH_PROFILER_DIR:
        logger.warning_once(
            "Torch Profiler is enabled in the API server. This should ONLY be "
            "used for local development!"
        )
    elif envs.VLLM_TORCH_CUDA_PROFILE:
        logger.warning_once(
            "CUDA Profiler is enabled in the API server. This should ONLY be "
            "used for local development!"
        )
    if envs.VLLM_TORCH_PROFILER_DIR or envs.VLLM_TORCH_CUDA_PROFILE:
        app.include_router(router)
