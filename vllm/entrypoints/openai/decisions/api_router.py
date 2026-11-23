# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)

from .protocol import DecisionRequest, DecisionResponse
from .serving import OpenAIServingDecisions

router = APIRouter()


@router.post(
    "/v1/decisions",
    dependencies=[Depends(validate_json_request)],
    response_model=DecisionResponse,
    responses={
        status.value: {"model": ErrorResponse}
        for status in (
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.NOT_FOUND,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.NOT_IMPLEMENTED,
        )
    },
)
@with_cancellation
@load_aware_call
async def create_decision(request: DecisionRequest, raw_request: Request):
    handler: OpenAIServingDecisions | None = getattr(
        raw_request.app.state, "openai_serving_decisions", None
    )
    if handler is None:
        error = create_error_response(
            NotImplementedError("The model does not support the Decisions API")
        )
        return JSONResponse(content=error.model_dump(), status_code=error.error.code)
    result = await handler.create_decisions(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())


def register_decisions_api_router(app: FastAPI):
    app.include_router(router)
