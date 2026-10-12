# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import AsyncGenerator
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.sse_obfuscation import (
    should_obfuscate,
    with_sse_obfuscation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def responses(request: Request) -> OpenAIServingResponses | None:
    return request.app.state.openai_serving_responses


async def _convert_stream_to_sse_events(
    generator: AsyncGenerator[StreamingResponsesResponse, None],
) -> AsyncGenerator[str, None]:
    """Convert the generator to a stream of events in SSE format."""
    async for event in generator:
        event_type = getattr(event, "type", "unknown")
        # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
        event_data = (
            f"event: {event_type}\ndata: "
            f"{event.model_dump_json(indent=None, by_alias=True)}\n\n"
        )
        yield event_data


def _sse_events(
    generator: AsyncGenerator[StreamingResponsesResponse, None],
    raw_request: Request,
    include_obfuscation: bool | None,
) -> AsyncGenerator[str, None]:
    args = getattr(raw_request.app.state, "args", None)
    enabled = should_obfuscate(
        include_obfuscation, getattr(args, "enable_stream_obfuscation", False)
    )
    # Upstream pads delta events only; other events are one-off.
    return with_sse_obfuscation(
        _convert_stream_to_sse_events(generator), enabled, event_suffix=".delta"
    )


@router.post(
    "/v1/responses",
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
async def create_responses(request: ResponsesRequest, raw_request: Request):
    handler = responses(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Responses API")

    generator = await handler.create_responses(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(mode="json", by_alias=True),
            status_code=generator.error.code,
        )
    elif isinstance(generator, ResponsesResponse):
        return JSONResponse(content=generator.model_dump(mode="json", by_alias=True))

    return StreamingResponse(
        content=_sse_events(
            generator,
            raw_request,
            getattr(request.stream_options, "include_obfuscation", None),
        ),
        media_type="text/event-stream",
    )


@router.get("/v1/responses/{response_id}")
@load_aware_call
async def retrieve_responses(
    response_id: str,
    raw_request: Request,
    starting_after: int | None = None,
    stream: bool | None = False,
    include_obfuscation: bool | None = None,
):
    handler = responses(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Responses API")

    response = await handler.retrieve_responses(
        response_id,
        starting_after=starting_after,
        stream=stream,
    )

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(mode="json", by_alias=True),
            status_code=response.error.code,
        )
    elif isinstance(response, ResponsesResponse):
        return JSONResponse(content=response.model_dump(mode="json", by_alias=True))
    return StreamingResponse(
        content=_sse_events(response, raw_request, include_obfuscation),
        media_type="text/event-stream",
    )


@router.post("/v1/responses/{response_id}/cancel")
@load_aware_call
async def cancel_responses(response_id: str, raw_request: Request):
    handler = responses(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Responses API")

    response = await handler.cancel_responses(response_id)

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(mode="json", by_alias=True),
            status_code=response.error.code,
        )
    return JSONResponse(content=response.model_dump(mode="json", by_alias=True))


def attach_router(app: FastAPI):
    app.include_router(router)
