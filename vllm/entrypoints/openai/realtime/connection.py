# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

import numpy as np
import pybase64 as base64
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.realtime.protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    RealtimeSessionConfig,
    SessionCreated,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)


def _holdback_rollback(text: str, tokenizer, rollback_tokens: int) -> str:
    """Return *text* with the last *rollback_tokens* tokens removed.

    This is the mirror of ``_rollback_prefix`` in the model file: it
    determines the "stable" portion of a segment's generation that is
    safe to send to the client.  The held-back tail will be re-decided
    by the next segment (or flushed at stream end).
    """
    if not text or rollback_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text)
    end_idx = max(0, len(token_ids) - rollback_tokens)
    if end_idx == 0:
        return ""
    stable = tokenizer.decode(token_ids[:end_idx])
    while "\ufffd" in stable and end_idx > 0:
        end_idx -= 1
        stable = tokenizer.decode(token_ids[:end_idx]) if end_idx > 0 else ""
    return stable


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
        self._session_config = RealtimeSessionConfig()

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
            model = event.get("model")
            err = self._check_model(model)
            if err is not None:
                await self.send_error(err.message, "model_not_found")
                return
            self._is_model_validated = True

            if event.get("language") is not None:
                self._session_config.language = event["language"]
            if event.get("prompt") is not None:
                self._session_config.prompt = event["prompt"]
            if event.get("segment_duration_s") is not None:
                self._session_config.segment_duration_s = float(
                    event["segment_duration_s"]
                )
            if event.get("rollback_tokens") is not None:
                self._session_config.rollback_tokens = int(
                    event["rollback_tokens"]
                )
            if event.get("unfixed_chunks") is not None:
                self._session_config.unfixed_chunks = int(
                    event["unfixed_chunks"]
                )
            if event.get("max_prefix_tokens") is not None:
                self._session_config.max_prefix_tokens = int(
                    event["max_prefix_tokens"]
                )
            if event.get("max_audio_s") is not None:
                self._session_config.max_audio_s = float(
                    event["max_audio_s"]
                )
            if event.get("realtime_max_tokens") is not None:
                self._session_config.realtime_max_tokens = int(
                    event["realtime_max_tokens"]
                )
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

        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()
        prefix_texts: deque[str] = deque()

        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream, input_stream, self._session_config,
            prefix_texts=prefix_texts,
        )

        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream,
                                 prefix_texts)
        )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
        prefix_texts: deque[str],
    ):
        """Run the generation and stream results back to the client.

        Each segment re-transcribes all accumulated audio.  The prompt
        includes a text *prefix* covering previously-confirmed text so
        the model only generates the *continuation*.  ``prefix_texts``
        carries the actual prefix string for each segment so we can
        reconstruct the full ``raw_decoded = prefix + gen_text`` and
        apply holdback identically to the model's rollback, keeping
        ``confirmed_text`` perfectly aligned.

        To avoid sending uncertain trailing text (e.g. "and." when the
        model doesn't yet have enough audio to know "and badly…"), we
        apply rollback holdback: only the *stable* portion of each
        segment's output is sent as a delta.  The trailing tokens that
        would be rolled back for the next segment are held back until
        either (a) the next segment confirms them, or (b) the stream
        ends and we flush everything.

        At segment end, a sentinel ``[]`` is pushed into ``input_stream``
        so ``buffer_realtime_audio`` can update its prefix for the next
        step.
        """
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        confirmed_text = ""

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0

        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            max_tokens = (self._session_config.realtime_max_tokens
                          or self.serving.model_cls.realtime_max_tokens)
            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            tokenizer = self.serving.renderer.get_tokenizer()

            from vllm.model_executor.models.qwen3_asr_realtime import (
                _DEFAULT_ROLLBACK_TOKENS,
            )
            rollback_tokens = (self._session_config.rollback_tokens
                               if self._session_config.rollback_tokens
                               is not None
                               else _DEFAULT_ROLLBACK_TOKENS)

            eos_ids: set[int] = set()
            try:
                mc = self.serving.model_config
                eos = getattr(mc.hf_config, "eos_token_id", None)
                if isinstance(eos, list):
                    eos_ids.update(eos)
                elif eos is not None:
                    eos_ids.add(eos)
                if not eos_ids:
                    gen_cfg = mc.try_get_generation_config()
                    if isinstance(gen_cfg, dict):
                        eos = gen_cfg.get("eos_token_id")
                    else:
                        eos = getattr(gen_cfg, "eos_token_id", None)
                    if isinstance(eos, list):
                        eos_ids.update(eos)
                    elif eos is not None:
                        eos_ids.add(eos)
            except Exception as e:
                logger.warning("Failed to get EOS token IDs: %s", e)

            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            segment_gen_text = ""
            prefix_text = ""
            segment_signalled = True
            last_prompt_len = 0

            async for output in result_gen:
                if not output.outputs or len(output.outputs) == 0:
                    continue

                cur_prompt_len = (len(output.prompt_token_ids)
                                  if output.prompt_token_ids else 0)
                new_segment_starting = (
                    cur_prompt_len > 0 and cur_prompt_len != last_prompt_len
                )

                if new_segment_starting:
                    last_prompt_len = cur_prompt_len
                    prompt_token_ids_len = cur_prompt_len
                    segment_gen_text = ""
                    segment_signalled = False
                    prefix_text = (prefix_texts.popleft()
                                   if prefix_texts else "")

                raw_delta = output.outputs[0].text
                tok_ids = list(output.outputs[0].token_ids)

                input_stream.put_nowait(tok_ids)
                completion_tokens_len += len(tok_ids)
                segment_gen_text += raw_delta

                has_eos = bool(eos_ids and eos_ids.intersection(tok_ids))
                if ((output.outputs[0].finish_reason is not None or has_eos)
                        and not segment_signalled):
                    segment_signalled = True

                    raw_decoded = (prefix_text
                                   + segment_gen_text.rstrip("\n").rstrip())
                    stable = _holdback_rollback(
                        raw_decoded, tokenizer, rollback_tokens)
                    if len(stable) > len(confirmed_text):
                        delta = stable[len(confirmed_text):]
                        confirmed_text = stable
                        await self.send(
                            TranscriptionDelta(delta=delta))

                    input_stream.put_nowait([])

                if not self._is_connected:
                    break

            # Flush remaining unstable text from the last segment now
            # that we know no more segments are coming.
            if segment_gen_text:
                raw_decoded = (prefix_text
                               + segment_gen_text.rstrip("\n").rstrip())
                if len(raw_decoded) > len(confirmed_text):
                    flush = raw_decoded[len(confirmed_text):]
                    confirmed_text += flush
                    await self.send(TranscriptionDelta(delta=flush))

            if not segment_signalled:
                input_stream.put_nowait([])

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            await self.send(TranscriptionDone(text=confirmed_text, usage=usage))

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
