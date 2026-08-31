# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import Response

from vllm.config import ProfilerConfig
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

# Regex to prevent path traversal: blocks slashes and parent directory sequences ('..')
_SECURE_FILENAME_REGEX = r"^(?!.*\.\.)[^/\\]*$"


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/start_profile")
async def start_profile(
    raw_request: Request,
    profile_prefix: Annotated[
        str | None,
        Query(
            pattern=_SECURE_FILENAME_REGEX,
            description="Prefix for start profiling. "
            "Validated to prevent path traversal.",
        ),
    ] = None,
):
    logger.info("Starting profiler...")
    await engine_client(raw_request).start_profile(profile_prefix=profile_prefix)
    logger.info("Profiler started.")
    return Response(status_code=200)


@router.post("/stop_profile")
async def stop_profile(raw_request: Request):
    logger.info("Stopping profiler...")
    await engine_client(raw_request).stop_profile()
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
