# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from vllm.engine.protocol import EngineClient

router = APIRouter()


class RuntimeConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_num_running_seqs: int = Field(ge=1, strict=True)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/v1/runtime_config")
async def get_runtime_config(raw_request: Request) -> dict[str, int]:
    """Return live scheduler policy and its startup-time capacity."""
    return await engine_client(raw_request).get_runtime_config()


@router.patch("/v1/runtime_config")
async def update_runtime_config(
    update: RuntimeConfigUpdate, raw_request: Request
) -> dict[str, int]:
    """Change scheduler policy without resizing engine resources."""
    engine = engine_client(raw_request)
    current_config = await engine.get_runtime_config()
    capacity = current_config.get("max_num_seqs_capacity")
    if capacity is None:
        raise HTTPException(
            status_code=501,
            detail="The configured scheduler does not support runtime configuration",
        )
    if update.max_num_running_seqs > capacity:
        raise HTTPException(
            status_code=400,
            detail=(
                "max_num_running_seqs cannot exceed the startup capacity "
                f"max_num_seqs ({capacity})"
            ),
        )
    return await engine.update_runtime_config(update.max_num_running_seqs)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
