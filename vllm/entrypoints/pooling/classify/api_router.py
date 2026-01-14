# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationRequest,
    ClassificationResponse,
)
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def classify(request: Request) -> ServingClassification | None:
    return request.app.state.openai_serving_classification


@router.post("/classify", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_classify(request: ClassificationRequest, raw_request: Request):
    handler = classify(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Classification API"
        )

    try:
        generator = await handler.create_classify(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ClassificationResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)
