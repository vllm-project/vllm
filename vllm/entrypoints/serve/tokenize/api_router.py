# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import (
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


router = APIRouter()


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

    try:
        generator = await handler.create_tokenize(request, raw_request)
    except Exception as e:
        return handler.create_error_response(e)

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

    try:
        generator = await handler.create_detokenize(request, raw_request)
    except OverflowError as e:
        raise RequestValidationError(errors=[str(e)]) from e
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

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
