# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
