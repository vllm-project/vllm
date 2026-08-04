# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, HTTPException, Request
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
    try:
        await engine_client(raw_request).start_profile()
    except Exception as exc:
        logger.exception("Failed to start profiler")
        raise HTTPException(
            status_code=500, detail=f"Failed to start profiler: {exc}"
        ) from exc
    logger.info("Profiler started.")
    return Response(status_code=200)


@router.post("/stop_profile")
async def stop_profile(raw_request: Request):
    logger.info("Stopping profiler...")
    try:
        await engine_client(raw_request).stop_profile()
    except Exception as exc:
        logger.exception("Failed to stop profiler")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop profiler: {exc}"
        ) from exc
    logger.info("Profiler stopped.")
    return Response(status_code=200)


def attach_router(app: FastAPI):
    profiler_config = getattr(app.state.args, "profiler_config", None)
    assert profiler_config is None or isinstance(profiler_config, ProfilerConfig)
    if profiler_config is not None and profiler_config.profiler is not None:
        logger.warning_once(
            "Profiler with mode '%s' is enabled in the "
            "API server. This should ONLY be used for local development!",
            profiler_config.profiler,
        )
        app.include_router(router)
