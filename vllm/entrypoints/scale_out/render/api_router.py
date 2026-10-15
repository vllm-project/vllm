# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    request_span,
    set_span_error,
    validate_json_request,
)
from vllm.logger import init_logger
from vllm.tracing import SpanAttributes

from ..token_in_token_out.protocol import GenerateRequest
from .serving import ServingRender

if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = init_logger(__name__)

router = APIRouter()
_tool_call_render_requests: Counter | None = None


def _tool_render_counter(
    request: ChatCompletionRequest | AnthropicMessagesRequest,
) -> Counter | None:
    if not any(
        field in request.model_fields_set and getattr(request, field, None) is not None
        for field in ("tools", "tool_choice", "parallel_tool_calls")
    ):
        return None

    global _tool_call_render_requests
    if _tool_call_render_requests is None:
        _tool_call_render_requests = Counter(
            "vllm:tool_call_render_requests_total",
            "Validated tool-field requests received by HTTP render endpoints, "
            "counted once per request independently of trace sampling. "
            "Rendered means rendering succeeded, not template equivalence.",
            ["model_name", "api_surface", "status"],
        )
    return _tool_call_render_requests


def _annotate_tool_render_span(
    request: ChatCompletionRequest | AnthropicMessagesRequest,
    span: "Span | None",
    api_surface: Literal["chat_completions", "messages"],
) -> None:
    if span is None or not span.is_recording():
        return
    choice = request.tool_choice
    choice_type = (
        choice if isinstance(choice, str) else getattr(choice, "type", "unset")
    )
    choice_label = {"any": "required", "tool": "named", "function": "named"}.get(
        choice_type, choice_type
    )
    if choice_label not in ("unset", "auto", "none", "required", "named"):
        choice_label = "other"
    span.set_attributes(
        {
            SpanAttributes.TOOL_CALLING_PRESENT: True,
            SpanAttributes.TOOL_CALLING_TOOL_COUNT: len(request.tools or ()),
            SpanAttributes.TOOL_CALLING_TOOL_CHOICE: choice_label,
            SpanAttributes.TOOL_CALLING_API_SURFACE: api_surface,
        }
    )


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

    counter = _tool_render_counter(request)
    status = "error"
    async with request_span(raw_request, "render_chat_completion") as span:
        if counter is not None:
            _annotate_tool_render_span(request, span, "chat_completions")
        try:
            result = await handler.render_chat_request(request, raw_request)
            if isinstance(result, ErrorResponse):
                set_span_error(span)
                return JSONResponse(
                    content=result.model_dump(), status_code=result.error.code
                )
            status = "rendered"
        finally:
            if counter is not None:
                counter.labels(
                    model_name=handler.model_config.served_model_name,
                    api_surface="chat_completions",
                    status=status,
                ).inc()
                if span is not None and span.is_recording():
                    span.set_attribute(
                        SpanAttributes.TOOL_CALLING_RENDER_STATUS, status
                    )

    return JSONResponse(content=result.model_dump())


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

    counter = _tool_render_counter(request)
    status = "error"
    async with request_span(raw_request, "render_messages") as span:
        if counter is not None:
            _annotate_tool_render_span(request, span, "messages")
        try:
            result = await handler.render_messages_request(request, raw_request)
            if isinstance(result, ErrorResponse):
                set_span_error(span)
                return JSONResponse(
                    content=result.model_dump(), status_code=result.error.code
                )
            status = "rendered"
        finally:
            if counter is not None:
                counter.labels(
                    model_name=handler.model_config.served_model_name,
                    api_surface="messages",
                    status=status,
                ).inc()
                if span is not None and span.is_recording():
                    span.set_attribute(
                        SpanAttributes.TOOL_CALLING_RENDER_STATUS, status
                    )

    return JSONResponse(content=result.model_dump())


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

    async with request_span(raw_request, "render_completion") as span:
        result = await handler.render_completion_request(request, raw_request)
        if isinstance(result, ErrorResponse):
            set_span_error(span)
            return JSONResponse(
                content=result.model_dump(), status_code=result.error.code
            )

    return JSONResponse(content=[item.model_dump() for item in result])


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

    async with request_span(raw_request, "render_responses") as span:
        result = await handler.render_responses_request(request)
        if isinstance(result, ErrorResponse):
            set_span_error(span)
            return JSONResponse(
                content=result.model_dump(), status_code=result.error.code
            )
    return JSONResponse(content=result.model_dump())
