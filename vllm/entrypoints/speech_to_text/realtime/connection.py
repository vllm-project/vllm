# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

import numpy as np
import pybase64 as base64
from fastapi import WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import RealtimeSegmentTimestamper

from .protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    SessionUpdate,
    SessionUpdated,
    TranscriptionDelta,
    TranscriptionDone,
    TranscriptionSegmentTimestamp,
)
from .serving import OpenAIServingRealtime

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
        self._is_model_validated = False
        self._segment_timestamps = False
        self._validation_error: str | None = None

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
                    await self.send_error(sanitize_message(str(e)), "processing_error")
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
            await self._handle_session_update(event)
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
                err_msg = self._validation_error or (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )
                return

            commit_event = InputAudioBufferCommit(**event)
            # final signals that the audio is finished
            if commit_event.final:
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def _reject_session_update(self, message: str, code: str) -> None:
        """Refuse a session.update and remember why.

        The session is left unconfigured so a later commit cannot run under a
        configuration the client did not get, and reports this reason instead
        of the generic "you never sent session.update".
        """
        self._is_model_validated = False
        self._validation_error = message
        await self.send_error(message, code)

    def _check_segment_timestamps(self) -> str | None:
        """Return why segment timestamps cannot be enabled, or None."""
        model_cls = self.serving.model_cls
        # `supports_realtime` is itself a duck-check, so an out-of-tree
        # realtime model need not define this attribute at all.
        if not getattr(model_cls, "supports_realtime_segment_timestamps", False):
            return (
                f"The model `{self.serving.model_config.model}` does not support "
                "segment timestamps. Omit `timestamp_granularities` from "
                "session.update, or serve a model that supports them."
            )

        vllm_config = getattr(self.serving.engine_client, "vllm_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        stream_interval = getattr(scheduler_config, "stream_interval", 1)
        if stream_interval > 1:
            return (
                f"Segment timestamps require `--stream-interval 1`, but this "
                f"server runs with {stream_interval}. Batched streaming stalls "
                "realtime transcription outright, because the next token "
                "cannot be generated until the previous one is streamed back, "
                "so omitting `timestamp_granularities` will not help. Restart "
                "the server with `--stream-interval 1`."
            )
        return None

    async def _handle_session_update(self, event: dict) -> None:
        """Validate the model and negotiate optional features."""
        if event.get("model") is None:
            await self._reject_session_update(
                "Missing required field: model", "invalid_event"
            )
            return

        try:
            session = SessionUpdate(**event)
        except ValidationError as e:
            if any("timestamp_granularities" in str(err["loc"]) for err in e.errors()):
                await self._reject_session_update(
                    "Only `segment` is a valid timestamp granularity on "
                    "/v1/realtime. `word` granularity is available on "
                    "POST /v1/audio/transcriptions.",
                    "invalid_timestamp_granularity",
                )
            else:
                await self._reject_session_update(
                    sanitize_message(str(e)), "invalid_event"
                )
            return

        err = self._check_model(session.model)
        if err is not None:
            await self._reject_session_update(err.error.message, "model_not_found")
            return

        # Assign rather than accumulate, so a later plain session.update turns
        # the feature back off.
        want_segments = "segment" in session.timestamp_granularities
        if want_segments:
            if (reason := self._check_segment_timestamps()) is not None:
                await self._reject_session_update(
                    reason, "unsupported_timestamp_granularity"
                )
                return
            try:
                # Built and discarded: surfaces a checkpoint whose tokenizer
                # cannot mark segments now, rather than mid-utterance. A model
                # advertising support without implementing it lands here too,
                # so catch that rather than unwinding to the connection loop
                # with the session left half-configured.
                self.serving.model_cls.get_realtime_segment_timestamper(
                    self.serving.model_config
                )
            except (ValueError, NotImplementedError, AttributeError) as e:
                # `raise NotImplementedError` carries no message of its own.
                reason = sanitize_message(str(e)) or (
                    f"The model `{self.serving.model_config.model}` cannot "
                    "build a segment timestamper."
                )
                await self._reject_session_update(
                    reason, "segment_timestamps_unavailable"
                )
                return

        self._segment_timestamps = want_segments
        self._is_model_validated = True
        self._validation_error = None

        await self.send(
            SessionUpdated(
                model=session.model,
                timestamp_granularities=["segment"] if want_segments else [],
            )
        )

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

        # One timestamper per utterance: each commit starts a new engine
        # request whose token index restarts at 0. Resolving it here also
        # freezes the event schema for this utterance, so a session.update
        # arriving mid-generation cannot change it.
        segment_timestamper: RealtimeSegmentTimestamper | None = None
        if self._segment_timestamps:
            segment_timestamper = (
                self.serving.model_cls.get_realtime_segment_timestamper(
                    self.serving.model_config
                )
            )

        # Start generation task
        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream, segment_timestamper)
        )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
        segment_timestamper: RealtimeSegmentTimestamper | None = None,
    ):
        """Run the generation and stream results back to the client.

        This method:
        1. Creates sampling parameters from session config
        2. Passes the streaming input generator to engine.generate()
        3. Streams transcription.delta events as text is generated
        4. Sends final transcription.done event with usage stats
        5. Feeds generated token IDs back to input_stream for next iteration
        6. Cleans up the audio queue

        When ``segment_timestamper`` is set, every event also carries the
        segments closed so far. The trailing segment is only known once
        generation ends, so it appears in ``transcription.done`` alone. On the
        exception path no ``done`` is sent, so no flush happens either and the
        trailing segment is dropped along with the rest of the utterance.
        """
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0
        all_segments: list[TranscriptionSegmentTimestamp] = []
        # Omit the field entirely for clients that did not opt in, so their
        # payloads stay byte-identical.
        exclude = None if segment_timestamper is not None else {"segments"}

        try:
            # Create sampling params
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=self.serving.model_cls.realtime_max_tokens,
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
                    completion = output.outputs[0]
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = completion.text
                    full_text += delta

                    segments: list[TranscriptionSegmentTimestamp] | None = None
                    if segment_timestamper is not None:
                        segments = [
                            TranscriptionSegmentTimestamp(text=text, end=end)
                            for text, end in segment_timestamper.process_token_ids(
                                completion.token_ids
                            )
                        ]
                        all_segments.extend(segments)

                    # append output to input
                    input_stream.put_nowait(list(completion.token_ids))
                    await self.send(
                        TranscriptionDelta(delta=delta, segments=segments),
                        exclude=exclude,
                    )

                    completion_tokens_len += len(completion.token_ids)

                if not self._is_connected:
                    # finish because websocket connection was killed
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            if segment_timestamper is not None:
                all_segments.extend(
                    TranscriptionSegmentTimestamp(text=text, end=end)
                    for text, end in segment_timestamper.flush()
                )

            # Send final completion event
            await self.send(
                TranscriptionDone(
                    text=full_text,
                    usage=usage,
                    segments=all_segments if segment_timestamper is not None else None,
                ),
                exclude=exclude,
            )

            # Clear queue for next utterance
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(sanitize_message(str(e)), "processing_error")

    async def send(
        self,
        event: SessionCreated | SessionUpdated | TranscriptionDelta | TranscriptionDone,
        exclude: set[str] | None = None,
    ):
        """Send event to client."""
        data = event.model_dump_json(exclude=exclude)
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
