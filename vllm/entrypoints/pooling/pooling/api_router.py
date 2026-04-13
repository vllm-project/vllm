# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.pooling.protocol import PoolingRequest
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def pooling(request: Request) -> ServingPooling | None:
    return request.app.state.serving_pooling


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
        raise NotImplementedError("The model does not support Pooling API")

    return await handler(request, raw_request)
