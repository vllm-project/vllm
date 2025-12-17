# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated, Literal

import pydantic
from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


router = APIRouter()
PydanticVllmConfig = pydantic.TypeAdapter(VllmConfig)


@router.get("/server_info")
async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], Query()] = "text",
):
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    server_info = {
        "vllm_config": str(vllm_config)
        if config_format == "text"
        else PydanticVllmConfig.dump_python(vllm_config, mode="json", fallback=str)
        # fallback=str is needed to handle e.g. torch.dtype
    }
    return JSONResponse(content=server_info)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
