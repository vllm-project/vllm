# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.get("/fault_tolerance/status")
async def get_fault_info(
    raw_request: Request,
):
    client = engine_client(raw_request)
    engine_status_dict = await client.get_fault_info()
    return JSONResponse(content=engine_status_dict)


def attach_router(app: FastAPI):
    app.include_router(router)
