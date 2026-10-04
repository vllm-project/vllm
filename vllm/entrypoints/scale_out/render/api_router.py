# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from pydantic import TypeAdapter

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

from ..token_in_token_out.protocol import GenerateRequest
from .serving import ServingRender

logger = init_logger(__name__)

router = APIRouter()
_generate_requests_adapter = TypeAdapter(list[GenerateRequest])


def render(request: Request) -> ServingRender | None:
    return getattr(request.app.state, "serving_render", None)


def _render_response(
    result: GenerateRequest | list[GenerateRequest] | ErrorResponse,
) -> Response:
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    content = (
        _generate_requests_adapter.dump_json(result)
        if isinstance(result, list)
        else result.model_dump_json()
    )
    return Response(content=content, media_type="application/json")


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

    result = await handler.render_chat_request(request)

    return _render_response(result)


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

    result = await handler.render_messages_request(request)

    return _render_response(result)


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

    result = await handler.render_completion_request(request)

    return _render_response(result)


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
    return _render_response(result)
