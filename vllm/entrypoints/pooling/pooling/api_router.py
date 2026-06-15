# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request

from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)

from .protocol import PoolingRequest
from .serving import ServingPooling


def pooling(request: Request) -> ServingPooling | None:
    return request.app.state.serving_pooling


@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Pooling API")

    return await handler(request, raw_request)
