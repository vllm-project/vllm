# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def models(request: Request) -> OpenAIServingModels | None:
    return getattr(request.app.state, "openai_serving_models", None)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)
    if handler is None:
        return JSONResponse(
            status_code=404, content={"error": "models endpoint not available"}
        )

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
