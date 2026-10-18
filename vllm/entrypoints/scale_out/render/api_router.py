# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.pooling.pooling.protocol import (
    GeneratePoolingRequest,
    IOProcessorRequest,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    create_json_response,
    validate_json_request,
)
from vllm.logger import init_logger

from ..token_in_token_out.protocol import GenerateRequest
from .serving import ServingRender

logger = init_logger(__name__)

router = APIRouter()


def render(request: Request) -> ServingRender | None:
    return getattr(request.app.state, "serving_render", None)


@router.post(
    "/v1/chat/completions/render",
    dependencies=[Depends(validate_json_request)],
    response_model=GenerateRequest,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support Chat Completions Render API"
        )

    result = await handler.render_chat_request(request, raw_request)

    return create_json_response(result)


@router.post(
    "/v1/messages/render",
    dependencies=[Depends(validate_json_request)],
    response_model=GenerateRequest,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_messages(request: AnthropicMessagesRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Messages Render API")

    result = await handler.render_messages_request(request, raw_request)

    return create_json_response(result)


@router.post(
    "/v1/completions/render",
    dependencies=[Depends(validate_json_request)],
    response_model=list[GenerateRequest],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_completion(request: CompletionRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions Render API")

    result = await handler.render_completion_request(request, raw_request)

    return create_json_response(result)


@router.post(
    "/v1/responses/render",
    dependencies=[Depends(validate_json_request)],
    response_model=GenerateRequest,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_responses(request: ResponsesRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Responses Render API")

    result = await handler.render_responses_request(request)
    return create_json_response(result)


@router.post(
    "/pooling/preprocess",
    dependencies=[Depends(validate_json_request)],
    response_model=list[GeneratePoolingRequest],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def preprocess_pooling(request: IOProcessorRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Pooling Preprocess API")

    result = await handler.preprocess_pooling_request(request, raw_request)

    return create_json_response(result)
