# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from functools import lru_cache
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

logger = init_logger(__name__)

_RESPONSE_CLASS_FOR_EMBEDDINGS = JSONResponse


@lru_cache(maxsize=1)
def try_load_orjson_response_class_for_embeddings():
    global _RESPONSE_CLASS_FOR_EMBEDDINGS
    if importlib.util.find_spec("orjson") is not None:
        from fastapi.responses import ORJSONResponse

        _RESPONSE_CLASS_FOR_EMBEDDINGS = ORJSONResponse
        return
    logger.warning_once(
        "To make v1/embeddings API fast, please install orjson by `pip install orjson`"
    )


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
        return handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, EmbeddingResponse):
        try_load_orjson_response_class_for_embeddings()
        return _RESPONSE_CLASS_FOR_EMBEDDINGS(content=generator.model_dump())
    elif isinstance(generator, EmbeddingBytesResponse):
        return StreamingResponse(
            content=generator.content,
            headers=generator.headers,
            media_type=generator.media_type,
        )

    assert_never(generator)
