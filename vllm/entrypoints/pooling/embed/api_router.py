# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import Request

from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)

from .protocol import CohereEmbedRequest, EmbeddingRequest
from .serving import ServingEmbedding


def embedding(request: Request) -> ServingEmbedding | None:
    return request.app.state.serving_embedding


@with_cancellation
@load_aware_call
async def create_embedding(
    request: EmbeddingRequest,
    raw_request: Request,
):
    handler = embedding(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Embeddings API")

    return await handler(request, raw_request)


@with_cancellation
@load_aware_call
async def create_cohere_embedding(
    request: CohereEmbedRequest,
    raw_request: Request,
):
    handler = embedding(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Embeddings API")

    return await handler(request, raw_request)
