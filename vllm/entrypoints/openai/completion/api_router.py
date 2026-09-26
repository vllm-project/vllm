# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
from itertools import chain
from math import isfinite

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
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


def _can_render_directly(response: CompletionResponse) -> bool:
    # JSON-mode serialization differs for arbitrary values and non-finite floats.
    if (
        response.kv_transfer_params is not None
        or response.ec_transfer_params is not None
        or response.metrics is not None
    ):
        return False
    models: list[BaseModel | None] = [
        response,
        response.usage,
        response.usage.prompt_tokens_details,
        response.usage.completion_tokens_details,
    ]
    for choice in response.choices:
        models.extend((choice, choice.logprobs))
        if choice.prompt_logprobs is not None:
            return False
        if (logprobs := choice.logprobs) is not None:
            if not all(map(isfinite, filter(None, logprobs.token_logprobs))):
                return False
            values = chain.from_iterable(
                map(dict.values, filter(None, logprobs.top_logprobs))
            )
            if not all(map(isfinite, values)):
                return False
    return not any(model.model_extra for model in models if model is not None)


class _CompletionJSONResponse(JSONResponse):
    def render(self, content: CompletionResponse) -> bytes:
        try:
            if _can_render_directly(content):
                return content.model_dump_json(warnings="error").encode("utf-8")
        except Exception:
            # Preserve legacy handling of invalid Unicode and mutated models.
            pass
        return super().render(content.model_dump())


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
        raise NotImplementedError("The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return _CompletionJSONResponse(
            content=generator,
            headers=metrics_header(metrics_header_format),
        )

    args = getattr(raw_request.app.state, "args", None)
    keep_alive_interval = getattr(args, "sse_keep_alive_interval", 0)
    return StreamingResponse(
        content=with_sse_keep_alive(generator, float(keep_alive_interval)),
        media_type="text/event-stream",
    )


def attach_router(app: FastAPI):
    app.include_router(router)
