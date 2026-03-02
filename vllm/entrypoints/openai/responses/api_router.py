# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import AsyncGenerator
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def responses(request: Request) -> OpenAIServingResponses | None:
    return request.app.state.openai_serving_responses


async def _convert_stream_to_sse_events(
    generator: AsyncGenerator[StreamingResponsesResponse, None],
) -> AsyncGenerator[str, None]:
    """Convert the generator to a stream of events in SSE format"""
    async for event in generator:
        event_type = getattr(event, "type", "unknown")
        # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
        event_data = (
            f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"
        )
        yield event_data


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
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Responses API"
        )
    try:
        generator = await handler.create_responses(request, raw_request)
    except Exception as e:
        generator = handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, ResponsesResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(
        content=_convert_stream_to_sse_events(generator), media_type="text/event-stream"
    )


@router.get("/v1/responses/{response_id}")
@load_aware_call
async def retrieve_responses(
    response_id: str,
    raw_request: Request,
    starting_after: int | None = None,
    stream: bool | None = False,
):
    handler = responses(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Responses API"
        )

    try:
        response = await handler.retrieve_responses(
            response_id,
            starting_after=starting_after,
            stream=stream,
        )
    except Exception as e:
        response = handler.create_error_response(e)

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )
    elif isinstance(response, ResponsesResponse):
        return JSONResponse(content=response.model_dump())
    return StreamingResponse(
        content=_convert_stream_to_sse_events(response), media_type="text/event-stream"
    )


@router.post("/v1/responses/{response_id}/cancel")
@load_aware_call
async def cancel_responses(response_id: str, raw_request: Request):
    handler = responses(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Responses API"
        )

    try:
        response = await handler.cancel_responses(response_id)
    except Exception as e:
        response = handler.create_error_response(e)

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )
    return JSONResponse(content=response.model_dump())


@router.websocket("/v1/responses")
async def create_responses_websocket(websocket: WebSocket):
    """WebSocket endpoint for Responses API WebSocket mode.

    Protocol:
    1. Client connects to ws://host/v1/responses
    2. Client sends {"type": "response.create", ...} messages
    3. Server streams response events as JSON text frames
    4. Client can use previous_response_id for continuation
    """
    import json

    app = websocket.app
    serving = app.state.openai_serving_responses

    await websocket.accept()

    if serving is None:
        await websocket.close(code=1008, reason="Responses API not supported")
        return

    # Connection limit check
    async with app.state.ws_responses_lock:
        if (
            app.state.ws_responses_active_connections
            >= app.state.ws_responses_max_connections
        ):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "status": HTTPStatus.TOO_MANY_REQUESTS,
                        "error": {
                            "code": "websocket_connection_limit_reached",
                            "message": "Maximum WebSocket connections reached",
                        },
                    }
                )
            )
            await websocket.close()
            return
        app.state.ws_responses_active_connections += 1

    try:
        from vllm.entrypoints.openai.responses.websocket import (
            WebSocketResponsesConnection,
        )

        connection = WebSocketResponsesConnection(websocket, serving)
        await connection.handle_connection()
    finally:
        async with app.state.ws_responses_lock:
            app.state.ws_responses_active_connections -= 1


def attach_router(app: FastAPI):
    app.include_router(router)
