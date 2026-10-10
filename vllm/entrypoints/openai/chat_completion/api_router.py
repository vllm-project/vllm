# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.entrypoints.serve.utils.sse_keep_alive import with_sse_keep_alive
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(validate_json_request)],
    # Register the streaming model alongside response_model's non-streaming model.
    # FastAPI also assigns it to application/json; openapi_extra restores the
    # non-streaming JSON schema after generation.
    responses={
        HTTPStatus.OK.value: {
            "model": ChatCompletionStreamResponse,
            "description": (
                "JSON response when stream=false. With stream=true, SSE data "
                "events contain a response chunk, an error, or [DONE]. "
                "Keep-alive comments are not data events."
            ),
            "content": {
                "text/event-stream": {
                    "itemSchema": {
                        "type": "object",
                        "required": ["data"],
                        "properties": {
                            "data": {
                                "type": "string",
                                "anyOf": [
                                    {"const": "[DONE]"},
                                    {
                                        "contentMediaType": "application/json",
                                        "contentSchema": {
                                            "$ref": (
                                                "#/components/schemas/"
                                                "ChatCompletionStreamResponse"
                                            )
                                        },
                                    },
                                    {
                                        "contentMediaType": "application/json",
                                        "contentSchema": {
                                            "$ref": "#/components/schemas/ErrorResponse"
                                        },
                                    },
                                ],
                            },
                        },
                    }
                },
            },
        },
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/ChatCompletionResponse"
                        }
                    }
                }
            }
        }
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    args = getattr(raw_request.app.state, "args", None)
    keep_alive_interval = getattr(args, "sse_keep_alive_interval", 0)
    return StreamingResponse(
        content=with_sse_keep_alive(generator, float(keep_alive_interval)),
        media_type="text/event-stream",
    )


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
