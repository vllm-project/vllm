# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP routes for the experimental streaming session API."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.streaming.protocol import SessionRequest
from vllm.entrypoints.openai.streaming.serving import OpenAIServingStreaming
from vllm.entrypoints.serve.utils.api_utils import with_cancellation
from vllm.renderers.params import ChatParams
from vllm.v1.streaming.session import validate_streaming_config

if TYPE_CHECKING:
    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient

router = APIRouter(prefix="/v1/streaming")
_MAX_FRAME_BYTES = 8 * 1024 * 1024


def _serving(request: Request) -> OpenAIServingStreaming:
    return request.app.state.openai_serving_streaming


@router.get("/config")
async def get_config(request: Request):
    return JSONResponse(_serving(request).config().model_dump())


@router.post("/sessions")
async def create_session(request: SessionRequest, raw_request: Request):
    resp = await _serving(raw_request).create_session(request)
    return JSONResponse(resp.model_dump())


async def _read_frame(raw_request: Request) -> bytes:
    try:
        declared = int(raw_request.headers.get("content-length", 0))
    except ValueError:
        declared = 0
    if declared > _MAX_FRAME_BYTES:
        raise HTTPException(413, f"frame body exceeds {_MAX_FRAME_BYTES} bytes")
    body = bytearray()
    async for chunk in raw_request.stream():
        if len(body) + len(chunk) > _MAX_FRAME_BYTES:
            raise HTTPException(413, f"frame body exceeds {_MAX_FRAME_BYTES} bytes")
        body.extend(chunk)
    return bytes(body)


@router.post("/sessions/{session_id}/frame")
@with_cancellation
async def push_frame(
    session_id: str,
    raw_request: Request,
    body: Annotated[bytes, Depends(_read_frame)],
):
    # The dependency consumes the bounded body before with_cancellation starts
    # listening for disconnects, so the two tasks never race to read it.
    resp = await _serving(raw_request).push_frame(session_id, body)
    return JSONResponse(resp.model_dump())


@router.delete("/sessions/{session_id}")
async def close_session(session_id: str, request: Request):
    resp = await _serving(request).close_session(session_id)
    return JSONResponse(resp.model_dump())


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def init_streaming_state(engine_client: "EngineClient", state: "State") -> None:
    validate_streaming_config(engine_client.vllm_config)
    if engine_client.vllm_config.parallel_config._api_process_count != 1:
        raise ValueError("--enable-streaming requires --api-server-count 1")
    online_renderer = state.online_renderer
    state.openai_serving_streaming = OpenAIServingStreaming(
        engine_client,
        chat_params=ChatParams(
            chat_template=online_renderer.chat_template,
            chat_template_content_format=online_renderer.chat_template_content_format,
            chat_template_kwargs={
                **online_renderer.default_chat_template_kwargs,
                "add_generation_prompt": True,
            },
        ),
    )
