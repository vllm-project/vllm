# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.sse_keep_alive import with_sse_keep_alive
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"
# When adding a new vLLM-specific field to ChatCompletionResponse,
# add it here as well.
_VLLM_CHAT_COMPLETION_FIELDS = frozenset(
    {
        "prompt_logprobs",
        "prompt_token_ids",
        "prompt_text",
        "kv_transfer_params",
        "ec_transfer_params",
        "metrics",
    }
)


def _serialize_chat_completion_response(
    response: ChatCompletionResponse,
    *,
    omit_unset_fields: bool,
) -> dict[str, Any]:
    content = response.model_dump()
    if omit_unset_fields:
        for field in _VLLM_CHAT_COMPLETION_FIELDS:
            if content.get(field) is None:
                content.pop(field, None)
    return content


def _omit_unset_fields(raw_request: Request) -> bool:
    args = getattr(raw_request.app.state, "args", None)
    return bool(getattr(args, "omit_unset_chat_completion_fields", False))


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
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
            content=_serialize_chat_completion_response(
                generator,
                omit_unset_fields=_omit_unset_fields(raw_request),
            ),
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

    return JSONResponse(
        content=_serialize_chat_completion_response(
            result,
            omit_unset_fields=_omit_unset_fields(raw_request),
        )
    )


def attach_router(app: FastAPI):
    app.include_router(router)
