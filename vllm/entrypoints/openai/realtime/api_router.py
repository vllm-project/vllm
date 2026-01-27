# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, WebSocket

from vllm.entrypoints.openai.realtime.connection import RealtimeConnection
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime audio transcription.

    Protocol:
    1. Client connects to ws://host/v1/realtime
    2. Server sends session.created event
    3. Client optionally sends session.update with model/params
    4. Client sends input_audio_buffer.append events with base64 PCM16 chunks
    5. Client sends input_audio_buffer.commit when ready
    6. Server processes and sends transcription.delta events
    7. Server sends transcription.done with final text + usage
    8. Repeat from step 4 for next utterance

    Audio format: PCM16, 16kHz, mono, base64-encoded
    """
    app = websocket.app
    serving = app.state.openai_serving_realtime

    if serving is None:
        logger.warning("Realtime transcription not supported - closing WebSocket")
        await websocket.close(code=1011, reason="Realtime transcription not supported")
        return

    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


def attach_router(app: FastAPI):
    """Attach the realtime router to the FastAPI app."""
    app.include_router(router)
    logger.info("Realtime API router attached")
