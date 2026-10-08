# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import json
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

from ...serve.engine.protocol import ErrorResponse
from .protocol import (
    GenerateRequest,
    GenerateResponseBase,
    RenderedGenerateResponse,
)
from .serving import ServingTokens

logger = init_logger(__name__)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def generate_tokens(request: Request) -> ServingTokens | None:
    return request.app.state.serving_tokens


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


class _RenderedJSONResponse(JSONResponse):
    """A body already rendered to JSON, sent with ``JSONResponse`` headers."""

    def render(self, content: Any) -> bytes:
        return content


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

    elif isinstance(generator, RenderedGenerateResponse):
        return _RenderedJSONResponse(content=generator.body)

    elif isinstance(generator, GenerateResponseBase):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


abort_router = APIRouter()


async def abort_requests(raw_request: Request):
    """Abort one or more requests. To be used in a
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
    asyncio.create_task(engine_client(raw_request).abort(request_ids))
    return Response(status_code=200)


# Under the `/inference` prefix, so `--api-key` guards it.
router.add_api_route("/inference/v1/abort_requests", abort_requests, methods=["POST"])
# Unauthenticated legacy path, kept for existing `--tokens-only` deployments.
abort_router.add_api_route("/abort_requests", abort_requests, methods=["POST"])


def attach_router(app: FastAPI):
    app.include_router(router)
    if not getattr(app.state.args, "tokens_only", False):
        return
    # The RLHF dev router registers its own /abort_requests first. Registering
    # this one too would only add a shadowed route with a duplicate operation ID.
    has_abort_route = any(
        getattr(route, "path", None) == "/abort_requests" for route in app.routes
    )
    if not has_abort_route:
        app.include_router(abort_router)
