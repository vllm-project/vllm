# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorResponse,
    PoolingBytesResponse,
    PoolingRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def pooling(request: Request) -> OpenAIServingPooling | None:
    return request.app.state.openai_serving_pooling


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
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Pooling API"
        )
    try:
        generator = await handler.create_pooling(request, raw_request)
    except Exception as e:
        return handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, (PoolingResponse, IOProcessorResponse)):
        return JSONResponse(content=generator.model_dump())
    elif isinstance(generator, PoolingBytesResponse):
        return StreamingResponse(
            content=generator.content,
            headers=generator.headers,
            media_type=generator.media_type,
        )

    assert_never(generator)
