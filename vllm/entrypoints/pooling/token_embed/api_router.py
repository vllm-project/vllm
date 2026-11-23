# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)

from .protocol import TokenEmbeddingRequest
from .serving import ServingTokenEmbedding

router = APIRouter()


def token_embedding(request: Request) -> ServingTokenEmbedding:
    return request.app.state.serving_token_embedding


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
async def create_token_embedding(
    request: TokenEmbeddingRequest,
    raw_request: Request,
):
    handler = token_embedding(raw_request)
    return await handler(request, raw_request)
