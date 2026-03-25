# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from vllm.entrypoints.gradient.protocol import GradientRequest
from vllm.entrypoints.gradient.serving import ServingGradient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def gradient_handler(request: Request) -> ServingGradient | None:
    return request.app.state.serving_gradient


@router.post(
    "/v1/gradients",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def compute_gradients(
    request: GradientRequest,
    raw_request: Request,
):
    handler = gradient_handler(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support the Gradients API. "
            "Gradient computation requires a text generation model."
        )

    return await handler(request, raw_request)
