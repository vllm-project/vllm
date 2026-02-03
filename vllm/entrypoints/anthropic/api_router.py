# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def messages(request: Request) -> AnthropicServingMessages:
    return request.app.state.anthropic_serving_messages


@router.post(
    "/v1/messages",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_messages(request: AnthropicMessagesRequest, raw_request: Request):
    def translate_error_response(response: ErrorResponse) -> JSONResponse:
        anthropic_error = AnthropicErrorResponse(
            error=AnthropicError(
                type=response.error.type,
                message=response.error.message,
            )
        )
        return JSONResponse(
            status_code=response.error.code, content=anthropic_error.model_dump()
        )

    handler = messages(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        error = base_server.create_error_response(
            message="The model does not support Messages API"
        )
        return translate_error_response(error)

    try:
        generator = await handler.create_messages(request, raw_request)
    except Exception as e:
        logger.exception("Error in create_messages: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message=str(e),
                )
            ).model_dump(),
        )

    if isinstance(generator, ErrorResponse):
        return translate_error_response(generator)

    elif isinstance(generator, AnthropicMessagesResponse):
        resp = generator.model_dump(exclude_none=True)
        logger.debug("Anthropic Messages Response: %s", resp)
        return JSONResponse(content=resp)

    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI):
    app.include_router(router)
