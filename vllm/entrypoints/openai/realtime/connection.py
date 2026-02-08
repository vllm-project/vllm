# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.realtime.protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)


class RealtimeConnection:
    """Manages WebSocket lifecycle and state for realtime transcription.

    This class handles:
    - WebSocket connection lifecycle (accept, receive, send, close)
    - Event routing (session.update, append, commit)
    - Audio buffering via asyncio.Queue
    - Generation task management
    - Error handling and cleanup
    """

    def __init__(self, websocket: WebSocket, serving: OpenAIServingRealtime):
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None

        self._is_connected = False
        self._is_input_finished = False
        self._is_model_validated = False

        self._max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.connection_id)
        self._is_connected = True

        # Send session created event
        await self.send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error(str(e), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        if self.serving._is_model_supported(model):
            return None

        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model
        - input_audio_buffer.append: Add audio chunk to queue
        - input_audio_buffer.commit: Start transcription generation
        """
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            self._check_model(event["model"])
            self._is_model_validated = True
        elif event_type == "input_audio_buffer.append":
            append_event = InputAudioBufferAppend(**event)
            try:
                audio_bytes = base64.b64decode(append_event.audio)
                # Convert PCM16 bytes to float32 numpy array
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if len(audio_array) / 1024**2 > self._max_audio_filesize_mb:
                    raise VLLMValidationError(
                        "Maximum file size exceeded",
                        parameter="audio_filesize_mb",
                        value=len(audio_array) / 1024**2,
                    )
                if len(audio_array) == 0:
                    raise VLLMValidationError("Can't process empty audio.")

                # Put audio chunk in queue
                self.audio_queue.put_nowait(audio_array)

            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self.send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            if not self._is_model_validated:
                err_msg = (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )

            commit_event = InputAudioBufferCommit(**event)
            # final signals that the audio is finished
            if commit_event.final:
                self._is_input_finished = True
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generator that yields audio chunks from the queue."""
        while True:
            audio_chunk = await self.audio_queue.get()
            if audio_chunk is None:  # Sentinel value to stop
                break
            yield audio_chunk

    async def start_generation(self):
        """Start the transcription generation task."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # Create audio stream generator
        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()

        # Transform to StreamingInput generator
        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream, input_stream
        )

        # Start generation task
        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream)
        )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Run the generation and stream results back to the client.

        This method:
        1. Creates sampling parameters from session config
        2. Passes the streaming input generator to engine.generate()
        3. Streams transcription.delta events as text is generated
        4. Sends final transcription.done event with usage stats
        5. Feeds generated token IDs back to input_stream for next iteration
        6. Cleans up the audio queue
        """
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0

        try:
            # Create sampling params
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=1,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            # Pass the streaming input generator to the engine
            # The engine will consume audio chunks as they arrive and
            # stream back transcription results incrementally
            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Stream results back to client as they're generated
            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    full_text += delta

                    # append output to input
                    input_stream.put_nowait(list(output.outputs[0].token_ids))
                    await self.send(TranscriptionDelta(delta=delta))

                    completion_tokens_len += len(output.outputs[0].token_ids)

                if not self._is_connected:
                    # finish because websocket connection was killed
                    break

                if self.audio_queue.empty() and self._is_input_finished:
                    # finish because client signals that audio input
                    # is finished
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            # Send final completion event
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            # Clear queue for next utterance
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")

    async def send(
        self, event: SessionCreated | TranscriptionDelta | TranscriptionDone
    ):
        """Send event to client."""
        data = event.model_dump_json()
        await self.websocket.send_text(data)

    async def send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        error_event = ErrorEvent(error=message, code=code)
        await self.websocket.send_text(error_event.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        # Signal audio stream to stop
        self.audio_queue.put_nowait(None)

        # Cancel generation task if running
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

        logger.debug("Connection cleanup complete: %s", self.connection_id)
