# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from vllm.config import ProfilerConfig
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class StartProfileRequest(BaseModel):
    """Optional overrides for one profiling session.

    Values are validated by the engine client.
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    profile_prefix: str | None = None
    delay_iterations: int | None = None
    max_iterations: int | None = None


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/start_profile")
async def start_profile(
    raw_request: Request, profile_request: StartProfileRequest | None = None
):
    logger.info("Starting profiler...")
    # Only pass supplied overrides so clients without them keep the
    # zero-argument call.
    overrides = profile_request.model_dump(exclude_none=True) if profile_request else {}
    await engine_client(raw_request).start_profile(**overrides)
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
