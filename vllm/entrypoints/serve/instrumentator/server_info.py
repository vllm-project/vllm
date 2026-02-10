# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import functools
from typing import Annotated, Literal

import pydantic
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.collect_env import get_env_info
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


router = APIRouter()
PydanticVllmConfig = pydantic.TypeAdapter(VllmConfig)


def _get_vllm_env_vars():
    from vllm.config.utils import normalize_value

    vllm_envs = {}
    for key in dir(envs):
        if key.startswith("VLLM_") and "KEY" not in key:
            value = getattr(envs, key, None)
            if value is not None:
                value = normalize_value(value)
                vllm_envs[key] = value
    return vllm_envs


@functools.lru_cache(maxsize=1)
def _get_system_env_info_cached():
    return get_env_info()._asdict()


@router.get("/server_info")
async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], Query()] = "text",
):
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    server_info = {
        "vllm_config": (
            str(vllm_config)
            if config_format == "text"
            else PydanticVllmConfig.dump_python(vllm_config, mode="json", fallback=str)
        ),
        # fallback=str is needed to handle e.g. torch.dtype
        "vllm_env": _get_vllm_env_vars(),
        "system_env": await asyncio.to_thread(_get_system_env_info_cached),
    }
    return JSONResponse(content=server_info)
