# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

from vllm.config import ProfilerConfig
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


@router.post("/start_mem_profile")
async def start_mem_profile(raw_request: Request):
    logger.info("Starting memory snapshot profiler...")
    await engine_client(raw_request).start_mem_profile()
    logger.info("Memory snapshot profiler started.")
    return Response(status_code=200)


@router.post("/stop_mem_profile")
async def stop_mem_profile(raw_request: Request):
    logger.info("Stopping memory snapshot profiler...")
    await engine_client(raw_request).stop_mem_profile()
    logger.info("Memory snapshot profiler stopped. Snapshots saved.")
    return Response(status_code=200)


def attach_router(app: FastAPI):
    profiler_config = getattr(app.state.args, "profiler_config", None)
    assert profiler_config is None or isinstance(profiler_config, ProfilerConfig)

    should_attach = False
    if profiler_config is not None:
        if profiler_config.profiler is not None:
            logger.warning_once(
                "Profiler with mode '%s' is enabled in the "
                "API server. This should ONLY be used for local development!",
                profiler_config.profiler,
            )
            should_attach = True

        if profiler_config.memory_profiler_enabled:
            logger.warning_once(
                "Memory snapshot profiler is enabled in the API server. "
                "Snapshots will be saved to '%s'. "
                "This should ONLY be used for local development!",
                profiler_config.memory_profiler_dir,
            )
            should_attach = True

    if should_attach:
        app.include_router(router)
