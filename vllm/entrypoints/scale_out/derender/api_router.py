# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

from ..token_in_token_out.protocol import (
    DerenderChatRequest,
    DerenderChatStreamRequest,
    DerenderCompletionRequest,
    DerenderCompletionStreamRequest,
)
from .serving import ServingDerender

logger = init_logger(__name__)

router = APIRouter()


def derender(request: Request) -> ServingDerender | None:
    return getattr(request.app.state, "serving_derender", None)


def _validation_error_response(exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        content={"detail": jsonable_encoder(exc.errors(include_url=False))},
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
    )


def _non_object_body_response() -> JSONResponse:
    return JSONResponse(
        content={"detail": "Request body must be a JSON object"},
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
    )


@router.post(
    "/v1/chat/completions/derender",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def derender_chat_completion(raw_request: Request):
    """Derender a generate response into a ChatCompletionResponse.

    Accepts both non-streaming (``stream=false``, default) and streaming
    (``stream=true``) request bodies on the same path.

    Non-streaming: body is ``DerenderChatRequest`` (``generate_response`` with
    the complete token list). Returns a ``ChatCompletionResponse``.

    Streaming: body is ``DerenderChatStreamRequest`` (one ``generate_chunk``
    delta + optional ``stream_state``). Returns
    ``{"chunk": <ChatCompletionStreamResponse>, "stream_state": <state>}``.
    The client carries ``stream_state`` between successive calls, one per SSE
    chunk from ``/inference/v1/generate``.
    """
    handler = derender(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support Chat Completions Derender API"
        )

    body = await raw_request.json()
    if not isinstance(body, dict):
        return _non_object_body_response()

    if body.get("stream", False):
        try:
            stream_request = DerenderChatStreamRequest.model_validate(body)
        except ValidationError as exc:
            return _validation_error_response(exc)
        stream_result = await handler.derender_chat_stream_response(stream_request)
        if isinstance(stream_result, ErrorResponse):
            return JSONResponse(
                content=stream_result.model_dump(),
                status_code=stream_result.error.code,
            )
        chunk, stream_state = stream_result
        return JSONResponse(
            content={
                "chunk": chunk.model_dump(),
                "stream_state": stream_state.model_dump(),
            }
        )

    try:
        request = DerenderChatRequest.model_validate(body)
    except ValidationError as exc:
        return _validation_error_response(exc)
    result = await handler.derender_chat_response(request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())


@router.post(
    "/v1/completions/derender",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def derender_completion(raw_request: Request):
    """Derender a generate response into a CompletionResponse.

    Accepts both non-streaming (``stream=false``, default) and streaming
    (``stream=true``) request bodies on the same path.

    Non-streaming: body is ``DerenderCompletionRequest``. Returns a
    ``CompletionResponse``.

    Streaming: body is ``DerenderCompletionStreamRequest`` (one
    ``generate_chunk`` + optional ``stream_state``). Returns
    ``{"chunk": <CompletionStreamResponse>, "stream_state": <state>}``.
    """
    handler = derender(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions Derender API")

    body = await raw_request.json()
    if not isinstance(body, dict):
        return _non_object_body_response()

    if body.get("stream", False):
        try:
            stream_request = DerenderCompletionStreamRequest.model_validate(body)
        except ValidationError as exc:
            return _validation_error_response(exc)
        stream_result = await handler.derender_completion_stream_response(
            stream_request
        )
        if isinstance(stream_result, ErrorResponse):
            return JSONResponse(
                content=stream_result.model_dump(),
                status_code=stream_result.error.code,
            )
        chunk, stream_state = stream_result
        return JSONResponse(
            content={
                "chunk": chunk.model_dump(),
                "stream_state": stream_state.model_dump(),
            }
        )

    try:
        request = DerenderCompletionRequest.model_validate(body)
    except ValidationError as exc:
        return _validation_error_response(exc)
    result = await handler.derender_completion_response(request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())
