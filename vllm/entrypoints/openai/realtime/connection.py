# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from uuid import uuid4

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.realtime.protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    SessionUpdate,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
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
        self.session_config: SessionUpdate | None = None
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.info("WebSocket connection accepted: %s", self.connection_id)

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
            logger.info("WebSocket disconnected: %s", self.connection_id)
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model, language, temperature
        - input_audio_buffer.append: Add audio chunk to queue
        - input_audio_buffer.commit: Start transcription generation
        """
        event_type = event.get("type")

        if event_type == "session.update":
            self.session_config = SessionUpdate(**event)
            logger.info("Session updated: %s", self.session_config.model_dump())

            # TODO: Validate model change
            # if self.session_config.model:
            #     await self.serving.validate_model(self.session_config.model)

        elif event_type == "input_audio_buffer.append":
            append_event = InputAudioBufferAppend(**event)
            try:
                audio_bytes = base64.b64decode(append_event.audio)
                # Convert PCM16 bytes to float32 numpy array
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # TODO: Add audio validation
                # if len(audio_array) == 0:
                #     raise ValueError("Empty audio chunk")
                # if not self._is_valid_audio(audio_array):
                #     raise ValueError("Invalid audio format")

                # Put audio chunk in queue
                await self.audio_queue.put(audio_array)

            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self.send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            await self.start_generation()

            # TODO: Add buffer clear event support
            # elif event_type == "input_audio_buffer.clear":
            #     await self.clear_audio_queue()

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

        # Transform to StreamingInput generator
        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream, self.session_config
        )

        # Start generation task
        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen)
        )

    async def _run_generation(self, streaming_input_gen: AsyncGenerator):
        """Run the generation and stream results back to the client.

        This method:
        1. Creates sampling parameters from session config
        2. Passes the streaming input generator to engine.generate()
        3. Streams transcription.delta events as text is generated
        4. Sends final transcription.done event with usage stats
        5. Cleans up the audio queue
        """
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""

        # TODO: Track metrics
        # start_time = time.time()
        # num_audio_chunks = 0

        try:
            # Create base sampling params (can be overridden per chunk in StreamingInput)
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            base_sampling_params = SamplingParams.from_optional(
                temperature=self.session_config.temperature
                if self.session_config
                else 0.0,
                max_tokens=1,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            # Pass the streaming input generator to the engine
            # The engine will consume audio chunks as they arrive and
            # stream back transcription results incrementally
            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=base_sampling_params,
                request_id=request_id,
            )

            # Stream results back to client as they're generated
            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    delta = output.outputs[0].text
                    full_text += delta
                    await self.send(TranscriptionDelta(delta=delta))

                    # TODO: Track token counts for usage stats
                    # completion_tokens += len(output.outputs[0].token_ids)

            # TODO: Calculate accurate usage stats
            usage = UsageInfo(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

            # Send final completion event
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            # TODO: Log metrics

            # Clear queue for next utterance
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")

    async def send(self, event: SessionCreated | TranscriptionDelta | TranscriptionDone):
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
        await self.audio_queue.put(None)

        # Cancel generation task if running
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass

        logger.info("Connection cleanup complete: %s", self.connection_id)
