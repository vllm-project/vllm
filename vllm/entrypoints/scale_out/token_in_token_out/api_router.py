# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

from .protocol import (
    GenerateRequest,
    GenerateResponse,
)
from .serving import ServingTokens

logger = init_logger(__name__)

# The event loop only keeps weak references to tasks, so a task whose only
# reference is the create_task() call can be collected before it finishes.
# Hold a strong reference until it completes.
_background_tasks: set[asyncio.Task] = set()


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def generate_tokens(request: Request) -> ServingTokens | None:
    return request.app.state.serving_tokens


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post(
    "/inference/v1/generate",
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
async def generate(request: GenerateRequest, raw_request: Request):
    handler = generate_tokens(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support generate tokens API")

    generator = await handler.serve_tokens(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, GenerateResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI):
    if getattr(app.state.args, "tokens_only", False):

        @router.post("/abort_requests")
        async def abort_requests(raw_request: Request):
            """
            Abort one or more requests. To be used in a
            Disaggregated Everything setup.
            """
            try:
                body = await raw_request.json()
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail=f"JSON decode error: {e}",
                ) from e
            request_ids = body.get("request_ids")
            if request_ids is None:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="Missing 'request_ids' in request body",
                )
            # Abort requests in background
            task = asyncio.create_task(engine_client(raw_request).abort(request_ids))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            return Response(status_code=200)

    app.include_router(router)
