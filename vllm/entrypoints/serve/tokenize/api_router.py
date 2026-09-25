# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import (
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


router = APIRouter()


def _tokenize_not_implemented() -> JSONResponse:
    """Some engines (e.g. diffusion-only serving) attach the tokenize router
    without wiring a tokenization handler. Return a controlled 501 instead of
    letting ``handler.create_tokenize`` raise ``AttributeError`` -> HTTP 500."""
    return JSONResponse(
        content=ErrorResponse(
            error=ErrorInfo(
                message="This engine does not support tokenization.",
                type="invalid_request_error",
                code=HTTPStatus.NOT_IMPLEMENTED.value,
            )
        ).model_dump(),
        status_code=HTTPStatus.NOT_IMPLEMENTED.value,
    )


@router.post(
    "/tokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)
    if handler is None:
        return _tokenize_not_implemented()

    generator = await handler.create_tokenize(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post(
    "/detokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)
    if handler is None:
        return _tokenize_not_implemented()

    generator = await handler.create_detokenize(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


def attach_router(app: FastAPI):
    if getattr(app.state.args, "enable_tokenizer_info_endpoint", False):
        """Conditionally register the tokenizer info endpoint if enabled."""

        @router.get("/tokenizer_info")
        async def get_tokenizer_info(raw_request: Request):
            """Get comprehensive tokenizer information."""
            result = await tokenization(raw_request).get_tokenizer_info()
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code
                if isinstance(result, ErrorResponse)
                else 200,
            )

    app.include_router(router)
