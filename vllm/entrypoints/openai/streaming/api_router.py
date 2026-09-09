# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastAPI routes for the REST streaming API."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from vllm.entrypoints.openai.streaming.protocol import SessionRequest
from vllm.entrypoints.openai.streaming.serving import (
    OpenAIServingStreaming,
    StreamingError,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
    from vllm.tasks import SupportedTask

router = APIRouter()

_PREFIX = "/v1/streaming"

# Upper bound on one frame's encoded body (a 4K JPEG is well under 4 MiB).
_MAX_FRAME_BYTES = 8 * 1024 * 1024


def _serving(request: Request) -> OpenAIServingStreaming:
    serving = getattr(request.app.state, "openai_serving_streaming", None)
    if serving is None:
        raise StreamingError(
            "streaming API is not enabled (requires a generate model)",
            status_code=501,
        )
    return serving


@router.get(_PREFIX + "/config")
async def get_config(request: Request):
    try:
        serving = _serving(request)
        resp = serving.config()
    except StreamingError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)
    return JSONResponse(resp.model_dump())


@router.post(_PREFIX + "/sessions")
async def create_session(request: Request):
    try:
        serving = _serving(request)
        req = SessionRequest(**(await request.json()))
        resp = await serving.create_session(req)
    except StreamingError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)
    except (ValidationError, ValueError, TypeError) as e:
        # Malformed JSON (json.JSONDecodeError is a ValueError), a non-dict
        # body (TypeError), pydantic validation, and SamplingParams bounds
        # are all client errors, not 500s.
        return JSONResponse({"error": f"invalid session config: {e}"}, status_code=400)
    return JSONResponse(resp.model_dump())


@router.post(_PREFIX + "/sessions/{session_id}/frame")
async def push_frame(session_id: str, request: Request):
    try:
        serving = _serving(request)
        # Cap the body BEFORE buffering it (and before any session lookup):
        # Content-Length first for the fast reject, then a counted streamed
        # read as the real guard (the header is client-controlled and absent
        # for chunked bodies).
        content_length = request.headers.get("content-length")
        try:
            declared = int(content_length) if content_length is not None else 0
        except ValueError:
            declared = 0
        if declared > _MAX_FRAME_BYTES:
            return JSONResponse(
                {"error": f"frame body exceeds {_MAX_FRAME_BYTES} bytes"},
                status_code=413,
            )
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > _MAX_FRAME_BYTES:
                return JSONResponse(
                    {"error": f"frame body exceeds {_MAX_FRAME_BYTES} bytes"},
                    status_code=413,
                )
        resp = await serving.push_frame(session_id, bytes(body))
    except StreamingError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)
    return JSONResponse(resp.model_dump())


@router.delete(_PREFIX + "/sessions/{session_id}")
async def close_session(session_id: str, request: Request):
    try:
        serving = _serving(request)
        resp = await serving.close_session(session_id)
    except StreamingError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)
    return JSONResponse(resp.model_dump())


def attach_router(app: FastAPI):
    """Attach the streaming router to the FastAPI app."""
    app.include_router(router)
    logger.info("Streaming API router attached")


def init_streaming_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: "RequestLogger | None",
    supported_tasks: tuple["SupportedTask", ...],
):
    """Attach an ``OpenAIServingStreaming`` instance to the app state.

    Sessions require token generation, so ``state.openai_serving_streaming``
    is left as ``None`` (routes answer 501) for non-generate models.
    """
    if "generate" not in supported_tasks:
        state.openai_serving_streaming = None
        return
    state.openai_serving_streaming = OpenAIServingStreaming(
        engine_client,
        request_logger=request_logger,
    )
