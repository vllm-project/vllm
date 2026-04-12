# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection handler for streaming video input."""

import asyncio
import base64
import io
import json
from http import HTTPStatus
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

try:
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:

    class ClientDisconnected(Exception):  # type: ignore[no-redef]
        """Placeholder when uvicorn not available."""
        pass

import contextlib

from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.video_realtime.protocol import (
    CompletionDelta,
    CompletionDone,
    ErrorEvent,
    GenerationTrigger,
    InputVideoBufferAppend,
    InputVideoBufferCommit,
    SessionCreated,
)
from vllm.entrypoints.openai.video_realtime.serving import (
    OpenAIServingRealtimeVideo,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_VIDEO_PROMPT = ""

DEFAULT_MAX_FRAMES_PER_COMMIT = 64

DEFAULT_VIDEO_BATCH_QUEUE_MAXSIZE = 4



def _decode_video_frame(payload_b64: str, fmt: str | None):
    """Decode base64 to a single frame (PIL Image) for multi_modal_data."""
    raw = base64.b64decode(payload_b64)
    if not raw:
        raise VLLMValidationError("Empty video frame data.")
    try:
        from PIL import Image
    except ImportError as err:
        raise VLLMValidationError(
            "PIL is required for streaming video. Install with: pip install Pillow"
        ) from err
    stream = io.BytesIO(raw)
    img = Image.open(stream).convert("RGB")
    return img


class RealtimeVideoConnection:
    """Manages WebSocket lifecycle for realtime video understanding.

    - Session: session.update (model, optional prompt)
    - Append: input_video_buffer.append (base64 frame, one per message)
    - Commit: input_video_buffer.commit (process buffer as one batch, optional final)
      - With frames: process video + prompt. With no frames: text-only turn (prompt only).
      - So "text-only" or "text first then streaming video" are supported.
    - Trigger: generation.trigger (set prompt to use for next generation)
    - Server sends: completion.delta, completion.done, error
    """

    def __init__(
        self,
        websocket: WebSocket,
        serving: OpenAIServingRealtimeVideo,
        *,
        max_frames_per_commit: int = DEFAULT_MAX_FRAMES_PER_COMMIT,
        video_batch_queue_maxsize: int = DEFAULT_VIDEO_BATCH_QUEUE_MAXSIZE,
    ):
        self.websocket = websocket
        self.connection_id = f"ws-video-{uuid4()}"
        self.serving = serving
        self._frame_buffer: list = []
        self._video_batch_queue: asyncio.Queue[list | None] = asyncio.Queue(
            maxsize=video_batch_queue_maxsize + 1
        )
        self._frame_idx_queue: asyncio.Queue[list | None] = asyncio.Queue(
            maxsize=video_batch_queue_maxsize + 1
        )
        self.last_frame_idx = 0
        self._video_batch_queue_maxsize = video_batch_queue_maxsize
        self.generation_task: asyncio.Task | None = None
        self._is_connected = False
        self._is_input_finished = False
        self._is_model_validated = False
        self._prompt_text = DEFAULT_VIDEO_PROMPT
        self._max_frames_per_commit = max_frames_per_commit
        self._prompt_consume_next = False
        self._is_trigger_generating = False

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket (video) connection accepted: %s", self.connection_id)
        self._is_connected = True
        await self._send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self._handle_event(event)
                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self._send_error(str(e), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self._cleanup()

    def _check_model(self, model: str | None) -> ErrorResponse | None:
        if self.serving._is_model_supported(model):
            return None
        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def _handle_event(self, event: dict):
        """Route events to handlers."""
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            self._check_model(event.get("model"))
            self._is_model_validated = True
            if "prompt" in event and event["prompt"]:
                self._prompt_text = event["prompt"]
        elif event_type == "input_video_buffer.append":
            append_evt = InputVideoBufferAppend(**event)
            try:
                frame = _decode_video_frame(append_evt.video, append_evt.format)
                self._frame_buffer.append(frame)
                if len(self._frame_buffer) > self._max_frames_per_commit:
                    raise VLLMValidationError(
                        f"Too many frames in buffer, max {self._max_frames_per_commit}"
                        "Send input_video_buffer.commit first."
                    )
            except Exception as e:
                logger.error("Failed to decode video frame: %s", e)
                await self._send_error("Invalid video data", "invalid_video")
        elif event_type == "input_video_buffer.commit":
            if not self._is_model_validated:
                await self._send_error(
                    "Model not validated. Send session.update with model first.",
                    "model_not_validated",
                )
                return
            commit_evt = InputVideoBufferCommit(**event)
            # Enqueue one batch: current buffer as list of frames, 
            # or empty list for text-only turn.
            # Use await put() so we block when queue is full (backpressure to client).
            if self._frame_buffer:
                await self._video_batch_queue.put(list(self._frame_buffer))
                frame_count = len(self._frame_buffer)
                frame_indices = list(
                    range(self.last_frame_idx, self.last_frame_idx + frame_count)
                )
                await self._frame_idx_queue.put(frame_indices)
                self.last_frame_idx += len(self._frame_buffer)
                self._frame_buffer = []
            else:
                await self._video_batch_queue.put([])
                await self._frame_idx_queue.put([])
            if commit_evt.final:
                self._is_input_finished = True
                await self._video_batch_queue.put(None)
                await self._frame_idx_queue.put(None)
            if self.generation_task is None or self.generation_task.done():
                await self._start_generation()
        elif event_type == "generation.trigger":
            # Trigger event: set prompt and mark as trigger generation
            trigger_evt = GenerationTrigger(**event)
            if trigger_evt.prompt is not None:
                self._prompt_text = trigger_evt.prompt
                self._prompt_consume_next = True
            logger.debug("Generation trigger set")

        else:
            await self._send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def _start_generation(self):
        """Start the generation loop: one engine.generate() per batch for real-time low latency."""  # noqa: E501
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        self.generation_task = asyncio.create_task(self._run_generation_loop())

    async def _run_generation_loop(self):
        """One generate per batch: 
        stream_video_realtime yields one StreamingInput per batch;
        we run one engine.generate() per yield for real-time understanding."""

        full_text = ""
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams
            while True:
                full_text = ""
                total_prompt_tokens = 0
                total_output_tokens = 0
                total_tokens = 0
                request_id = f"rt-video-{self.connection_id}-{uuid4()}"
                stream_gen = self.serving.stream_video_realtime(
                    self._video_batch_queue,
                    self._frame_idx_queue,
                    prompt_getter=lambda: self._get_prompt_with_consume(),
                )

                max_tokens = 512

                sampling_params = SamplingParams.from_optional(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    output_kind=RequestOutputKind.DELTA,
                    skip_clone=True,
                )
                result_gen = self.serving.engine_client.generate(
                    prompt=stream_gen,
                    sampling_params=sampling_params,
                    request_id=request_id,
                )
                max_model_len = self.serving.engine_client.model_config.max_model_len
                async for output in result_gen:
                    if output.outputs and len(output.outputs) > 0:
                        if output.prompt_token_ids:
                            num_prompt_tokens = len(output.prompt_token_ids)
                        else:
                            num_prompt_tokens = 0
                        total_prompt_tokens += num_prompt_tokens
                        logger.info('input prompt token: %d', num_prompt_tokens) 
                        delta = output.outputs[0].text
                        full_text += delta
                        await self._send(CompletionDelta(delta=delta))
                        num_output_tokens = len(output.outputs[0].token_ids)
                        logger.info('output tokens: %d', num_output_tokens)
                        total_output_tokens += num_output_tokens
                        total_tokens = total_prompt_tokens + total_output_tokens
                        if total_tokens > max_model_len//2:
                            logger.info('starting truncation from connection side')
                            break  

                    if not self._is_connected:
                        break
                if not self._is_connected:
                    break
            
            usage = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_output_tokens,
                total_tokens=total_tokens
            )

            await self._send(
                CompletionDone(
                    text=full_text,
                    usage=usage
                )
            )

        except asyncio.CancelledError:
            self._is_trigger_generating = False
            pass
        except (WebSocketDisconnect, ClientDisconnected):
            self._is_trigger_generating = False
            logger.debug(
                "Video generation stopped: client disconnected (%s)",
                self.connection_id,
            )
        except Exception as e:
            self._is_trigger_generating = False
            logger.exception("Error in video generation: %s", e)
            with contextlib.suppress(WebSocketDisconnect, ClientDisconnected):
                await self._send_error(str(e), "processing_error")

    def _get_prompt_with_consume(self) -> str:
        if self._prompt_consume_next:
            self._prompt_consume_next = False
            prompt = self._prompt_text
            self._prompt_text = ""
            self._is_trigger_generating = True
            return prompt
        return ""

    async def _send(self, event):
        """Send event to client. Sets _is_connected=False on disconnect."""
        try:
            await self.websocket.send_text(event.model_dump_json())
        except (WebSocketDisconnect, ClientDisconnected) as e:
            self._is_connected = False
            logger.debug(
                "WebSocket closed while sending (connection_id=%s): %s",
                self.connection_id,
                e,
            )
            raise

    async def _send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        await self.websocket.send_text(
            ErrorEvent(error=message, code=code).model_dump_json()
        )

    async def _cleanup(self):
        """Cleanup resources. Awaits generation_task after cancel so shutdown doesn't hang."""
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self.generation_task, timeout=10.0)
        # Drain queue so we have room for None; then signal consumer to exit.
        while not self._video_batch_queue.empty():
            try:
                self._video_batch_queue.get_nowait()
                self._frame_idx_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self._video_batch_queue.put_nowait(None)
            self._frame_idx_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        logger.debug("Connection cleanup complete: %s", self.connection_id)
