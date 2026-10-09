# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)

from .protocol import StructuredDecisionRequest
from .serving import ServingStructuredDecisions

router = APIRouter()


def structured_decisions(request: Request) -> ServingStructuredDecisions | None:
    return getattr(request.app.state, "serving_structured_decisions", None)


@router.post(
    "/v1/systemone",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_structured_decision(
    request: StructuredDecisionRequest, raw_request: Request
):
    handler = structured_decisions(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support the structured decision API"
        )
    result = await handler.create_decision(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())


def register_structured_decisions_api_router(app: FastAPI):
    app.include_router(router)
