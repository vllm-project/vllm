# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.utils import (
    create_error_response,
    load_aware_call,
    with_cancellation,
)

router = APIRouter()


def classify(request: Request) -> ServingClassification | None:
    return request.app.state.serving_classification


@router.post("/classify", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_classify(
    request: ClassificationRequest, raw_request: Request
) -> Response:
    handler = classify(raw_request)
    if handler is None:
        error_response = create_error_response(
            message="The model does not support Classification API"
        )
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=error_response.error.code,
        )

    return await handler(request, raw_request)
