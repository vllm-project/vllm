# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/set_slowdown_threshold")
async def set_slowdown_threshold(raw_request: Request):
    threshold = int(raw_request.query_params.get("threshold", "0"))
    logger.info("Setting slowdown threshold to %d requests.", threshold)
    await engine_client(raw_request).set_slowdown_threshold(threshold)
    return Response(status_code=200)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)

