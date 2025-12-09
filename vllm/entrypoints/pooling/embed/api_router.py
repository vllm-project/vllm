# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def embedding(request: Request) -> OpenAIServingEmbedding | None:
    return request.app.state.openai_serving_embedding


@router.post(
    "/v1/embeddings",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_embedding(
    request: EmbeddingRequest,
    raw_request: Request,
):
    handler = embedding(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Embeddings API"
        )

    try:
        generator = await handler.create_embedding(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())
    elif isinstance(generator, EmbeddingBytesResponse):
        return StreamingResponse(
            content=generator.content,
            headers=generator.headers,
            media_type=generator.media_type,
        )

    assert_never(generator)
