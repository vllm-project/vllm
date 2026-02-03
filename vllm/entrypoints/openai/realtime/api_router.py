# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket

from vllm.entrypoints.openai.realtime.connection import RealtimeConnection
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object

router = APIRouter()


@router.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime audio transcription.

    Protocol:
    1. Client connects to ws://host/v1/realtime
    2. Server sends session.created event
    3. Client optionally sends session.update with model/params
    4. Client sends input_audio_buffer.commit when ready
    5. Client sends input_audio_buffer.append events with base64 PCM16 chunks
    6. Server processes and sends transcription.delta events
    7. Server sends transcription.done with final text + usage
    8. Repeat from step 5 for next utterance
    9. Optionally, client sends input_audio_buffer.commit with final=True
       to signal audio input is finished. Useful when streaming audio files

    Audio format: PCM16, 16kHz, mono, base64-encoded
    """
    app = websocket.app
    serving = app.state.openai_serving_realtime

    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


def attach_router(app: FastAPI):
    """Attach the realtime router to the FastAPI app."""
    app.include_router(router)
    logger.info("Realtime API router attached")


def init_realtime_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    state.openai_serving_realtime = (
        OpenAIServingRealtime(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
        )
        if "realtime" in supported_tasks
        else None
    )
