# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import time
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.request_log.aggregate import (
    aggregate_completion_stream,
)
from vllm.entrypoints.openai.request_log.client import (
    make_record,
    stream_logging_wrapper,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


@router.post(
    "/v1/completions",
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
async def create_completion(request: CompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Completions API"
        )

    received_at = time.time()
    request_log_hub = getattr(raw_request.app.state, "request_log_hub", None)
    try:
        generator = await handler.create_completion(request, raw_request)
    except Exception as e:
        generator = handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        if request_log_hub is not None:
            request_log_hub.log(
                make_record(
                    raw_request=raw_request,
                    endpoint="/v1/completions",
                    request_obj=request,
                    response=None,
                    received_at=received_at,
                    error=generator.model_dump(),
                )
            )
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        if request_log_hub is not None:
            request_log_hub.log(
                make_record(
                    raw_request=raw_request,
                    endpoint="/v1/completions",
                    request_obj=request,
                    response=generator.model_dump(),
                    received_at=received_at,
                )
            )
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    if request_log_hub is not None:
        generator = stream_logging_wrapper(
            generator,
            hub=request_log_hub,
            aggregator=aggregate_completion_stream,
            raw_request=raw_request,
            endpoint="/v1/completions",
            request_obj=request,
            received_at=received_at,
        )
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/completions/render",
    dependencies=[Depends(validate_json_request)],
    response_model=list,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_completion(request: CompletionRequest, raw_request: Request):
    """render completion request and return engine prompts without generating."""
    handler = completion(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Completions API"
        )

    try:
        result = await handler.render_completion_request(request)
    except Exception as e:
        result = handler.create_error_response(e)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result)


def attach_router(app: FastAPI):
    app.include_router(router)
