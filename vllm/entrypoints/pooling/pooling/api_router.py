# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)

from .protocol import GeneratePoolingRequest, GeneratePoolingResponse, PoolingRequest
from .serving import ServingPooling

router = APIRouter()


def pooling(request: Request) -> ServingPooling:
    handler = getattr(request.app.state, "serving_pooling", None)
    if handler is None:
        raise NotImplementedError("The model does not support Pooling API")
    return handler


@router.post(
    "/pooling",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    return await handler(request, raw_request)


@router.post(
    "/pooling/inference",
    dependencies=[Depends(validate_json_request)],
    response_model=GeneratePoolingResponse,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def pooling_inference(request: GeneratePoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support generate pooling API")

    generator = await handler.serve_pooling_embeds(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    return JSONResponse(content=generator.model_dump())
