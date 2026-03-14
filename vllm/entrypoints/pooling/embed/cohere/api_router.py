# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.embed.cohere.protocol import CohereEmbedRequest
from vllm.entrypoints.pooling.embed.cohere.serving import (
    CohereServingEmbedding,
)
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def cohere_embedding(request: Request) -> CohereServingEmbedding | None:
    return getattr(request.app.state, "cohere_serving_embedding", None)


@router.post(
    "/v2/embed",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_cohere_embedding(
    request: CohereEmbedRequest,
    raw_request: Request,
):
    handler = cohere_embedding(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Embeddings API")

    return await handler.create_embedding(request, raw_request)
