# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
from typing import cast

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.entrypoints.utils import (
    get_max_tokens,
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
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
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Chat Completions API"
        )

    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        generator = handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/render",
    dependencies=[Depends(validate_json_request)],
    response_model=GenerateRequest,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Render chat completion request into a token-in GenerateRequest.

    This endpoint must be a pure preprocessing step: it should not generate
    text and must return a JSON-serializable structure.
    """
    handler = chat(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Chat Completions API"
        )

    try:
        result = await handler.render_chat_request(request)
    except Exception as e:
        result = handler.create_error_response(e)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    _, engine_prompts = result

    if not engine_prompts:
        err = handler.create_error_response(ValueError("No engine prompts rendered"))
        return JSONResponse(content=err.model_dump(), status_code=err.error.code)

    if len(engine_prompts) != 1:
        err = handler.create_error_response(
            ValueError("Multiple engine prompts are not supported by this endpoint")
        )
        return JSONResponse(content=err.model_dump(), status_code=err.error.code)

    engine_prompt = engine_prompts[0]

    # Extract token IDs only; do not return internal multimodal artifacts.
    prompt_components = handler._extract_prompt_components(engine_prompt)
    token_ids = prompt_components.token_ids
    if not token_ids:
        err = handler.create_error_response(ValueError("No token_ids rendered"))
        return JSONResponse(content=err.model_dump(), status_code=err.error.code)
    token_ids = cast(list[int], token_ids)

    # Build sampling params using the same logic as generation.
    max_model_len = handler.model_config.max_model_len
    max_tokens = get_max_tokens(
        max_model_len,
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens,
        handler._extract_prompt_len(engine_prompt),
        handler.default_sampling_params,
        handler.override_max_tokens,
    )

    if request.use_beam_search:
        err = handler.create_error_response(
            ValueError(
                "Beam search is not supported by the token-in render endpoint"
            )
        )
        return JSONResponse(content=err.model_dump(), status_code=err.error.code)

    sampling_params: SamplingParams = request.to_sampling_params(
        max_tokens,
        handler.default_sampling_params,
    )

    # Prefix matches OpenAI-style Chat Completions IDs (e.g. "chatcmpl-...")
    # to keep logs/tooling consistent with /v1/chat/completions.
    request_id = f"chatcmpl-{handler._base_request_id(raw_request, request.request_id)}"

    # NOTE: 'features' is a placeholder in the RFC. Multimodal inputs are
    # currently rendered internally; we avoid returning non-JSON-serializable
    # tensors here.
    return GenerateRequest(
        request_id=request_id,
        token_ids=token_ids,
        sampling_params=sampling_params,
        model=request.model,
        # Preserve stream intent on the returned token-in request. The /render
        # HTTP response itself is always non-streamed JSON.
        stream=bool(request.stream),
        stream_options=request.stream_options if request.stream else None,
        cache_salt=request.cache_salt,
        priority=request.priority,
    )


def attach_router(app: FastAPI):
    app.include_router(router)
