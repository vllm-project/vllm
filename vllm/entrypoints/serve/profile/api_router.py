# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from vllm.config import ProfilerConfig
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class StartProfileRequest(BaseModel):
    num_steps: int | None = Field(
        default=None,
        description="Maximum number of engine steps to profile. "
        "Overrides the server's max_iterations config for this session. "
        "0 means unlimited.",
    )
    delay_steps: int | None = Field(
        default=None,
        description="Number of engine steps to skip before profiling begins. "
        "Overrides the server's delay_iterations config for this session.",
    )


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/start_profile")
async def start_profile(raw_request: Request, body: StartProfileRequest | None = None):
    if body is None:
        body = StartProfileRequest()
    logger.info(
        "Starting profiler... (num_steps=%s, delay_steps=%s)",
        body.num_steps,
        body.delay_steps,
    )
    await engine_client(raw_request).start_profile(
        num_steps=body.num_steps,
        delay_steps=body.delay_steps,
    )
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
