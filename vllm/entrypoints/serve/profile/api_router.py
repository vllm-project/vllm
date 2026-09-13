# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vllm.config import ProfilerConfig
from vllm.config.profiler import validate_profile_prefix
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class StartProfileRequest(BaseModel):
    """Per-session profiling overrides."""

    model_config = ConfigDict(extra="forbid", strict=True)

    profile_prefix: str | None = None
    delay_iterations: int | None = Field(default=None, ge=0)
    max_iterations: int | None = Field(default=None, ge=0)

    @field_validator("profile_prefix")
    @classmethod
    def validate_prefix(cls, value: str | None) -> str | None:
        if value is not None:
            validate_profile_prefix(value)
        return value


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/start_profile")
async def start_profile(
    raw_request: Request, profile_request: StartProfileRequest | None = None
):
    logger.info("Starting profiler...")
    if profile_request is None:
        await engine_client(raw_request).start_profile()
    else:
        await engine_client(raw_request).start_profile(
            profile_request.profile_prefix,
            profile_request.delay_iterations,
            profile_request.max_iterations,
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
