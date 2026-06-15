# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.disagg.protocol import (
    DerenderChatRequest,
    DerenderCompletionRequest,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.logger import init_logger

logger = init_logger(__name__)


def render(request: Request) -> OpenAIServingRender | None:
    return getattr(request.app.state, "openai_serving_render", None)


async def render_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support Chat Completions Render API"
        )

    result = await handler.render_chat_request(request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


async def render_completion(request: CompletionRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions Render API")

    result = await handler.render_completion_request(request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=[item.model_dump() for item in result])


async def derender_chat_completion(request: DerenderChatRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support Chat Completions Derender API"
        )

    result = await handler.derender_chat_response(request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


async def derender_completion(request: DerenderCompletionRequest, raw_request: Request):
    handler = render(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions Derender API")

    result = await handler.derender_completion_response(request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())
