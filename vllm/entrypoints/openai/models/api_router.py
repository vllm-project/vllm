# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.utils import create_error_response
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


# `{model_id:path}` accepts forward slashes so HuggingFace-style IDs
# (e.g. `meta-llama/Llama-3.1-8B-Instruct`) match without URL-encoding.
@router.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, raw_request: Request):
    handler = models(raw_request)

    card = await handler.retrieve_model(model_id)
    if card is None:
        error = create_error_response(
            message=f"The model `{model_id}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content=error.model_dump(),
        )
    return JSONResponse(content=card.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
