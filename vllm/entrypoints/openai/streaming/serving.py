# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSE serving for continuous video-stream captioning.

Per segment from the persistent DeepStream pipeline: build a video prompt,
run inference, emit one ``chat.completion.chunk``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy.typing as npt
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.multimodal.rtsp_stream_manager import (
    DEFAULT_CHUNK_DURATION,
    DEFAULT_NUM_FRAMES,
    RTSPStreamManager,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = init_logger(__name__)

# How often to wake while waiting for a segment, to check for disconnect.
_DISCONNECT_POLL_INTERVAL = 1.0

# Optional local video used to warm the pipeline at startup.
_PREWARM_ENV_VAR = "VLLM_RTSP_PREWARM_VIDEO"


def _extract_user_text(messages) -> str:
    """Pull concatenated text from the most recent user message."""
    default = "Describe what is happening in this video segment."
    for msg in reversed(list(messages)):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, str):
            return content or default
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                else:
                    if getattr(part, "type", None) == "text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            texts.append(text)
            if texts:
                return "\n".join(texts)
        return default
    return default


class VideoStreamingServing(GenerateBaseServing):
    """Continuous video-stream captioning via SSE."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )
        self._stream_manager = RTSPStreamManager()

    def _build_video_prompt(self, user_text: str) -> str:
        """Apply the model's chat template for a single-turn video+text message."""
        tokenizer = self.renderer.tokenizer
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    async def prewarm(
        self,
        sample_path: str | None = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
    ) -> None:
        """Warm the pipeline from a local video, if one is configured."""
        sample_path = sample_path or os.environ.get(_PREWARM_ENV_VAR)
        if not sample_path:
            return
        if not os.path.exists(sample_path):
            logger.warning(
                "Prewarm sample video not found at %s, skipping pipeline pre-warm",
                sample_path,
            )
            return

        uri = f"file://{sample_path}"
        logger.info("Pre-warming pipeline with %s", sample_path)
        consumer_id, seg_queue = await self._stream_manager.subscribe(
            uri=uri,
            chunk_duration=chunk_duration,
            num_frames=num_frames,
        )

        async def _drain() -> None:
            try:
                while True:
                    item = await seg_queue.get()
                    if item is None:
                        break
            finally:
                await self._stream_manager.unsubscribe(
                    uri=uri,
                    chunk_duration=chunk_duration,
                    num_frames=num_frames,
                    consumer_id=consumer_id,
                )
            logger.info("Pipeline prewarm complete")

        asyncio.create_task(_drain(), name="pipeline-prewarm")

    async def create_video_chat_stream(
        self,
        request,
        raw_request: Request,
        rtsp_url: str,
    ) -> AsyncGenerator[str, None] | ErrorResponse:
        """Validate and return a chat.completion.chunk SSE generator."""
        if not self._is_model_supported(request.model):
            return self.create_error_response(
                message=f"The model {request.model!r} is not available.",
            )
        return self._stream_chat_segments(request, raw_request, rtsp_url)

    async def _stream_chat_segments(
        self,
        request,
        raw_request: Request,
        rtsp_url: str,
    ) -> AsyncGenerator[str, None]:
        """Yield ``data: {chat.completion.chunk}\\n\\n`` SSE events."""
        extras = getattr(request, "model_extra", None) or {}
        chunk_duration_raw = extras.get("chunk_duration")
        try:
            chunk_duration = (
                DEFAULT_CHUNK_DURATION
                if chunk_duration_raw is None
                else float(chunk_duration_raw)
            )
        except (TypeError, ValueError):
            chunk_duration = DEFAULT_CHUNK_DURATION
        num_frames_raw = extras.get("num_frames_per_chunk")
        if num_frames_raw is None:
            num_frames_raw = extras.get("num_frames", DEFAULT_NUM_FRAMES)
        try:
            num_frames = int(num_frames_raw)
        except (TypeError, ValueError):
            num_frames = DEFAULT_NUM_FRAMES

        prompt_text = _extract_user_text(request.messages)
        request_id = f"chatcmpl-{random_uuid()}"
        created = int(time.time())
        model_name = request.model or ""

        consumer_id: str | None = None
        try:
            consumer_id, segment_queue = await self._stream_manager.subscribe(
                uri=rtsp_url,
                chunk_duration=chunk_duration,
                num_frames=num_frames,
            )

            while True:
                # Poll: a live source may go quiet, and a disconnect has to
                # be noticed so the pipeline is released.
                try:
                    segment = await asyncio.wait_for(
                        segment_queue.get(), timeout=_DISCONNECT_POLL_INTERVAL
                    )
                except asyncio.TimeoutError:
                    if await raw_request.is_disconnected():
                        break
                    continue

                if segment is None:
                    break
                if await raw_request.is_disconnected():
                    break

                frames: npt.NDArray = segment[0]
                metadata: dict[str, Any] = segment[1]
                seg_request_id = f"vseg-{random_uuid()}"

                try:
                    caption = await self._infer_chat_caption(
                        frames,
                        request,
                        prompt_text,
                        seg_request_id,
                    )
                except Exception:
                    logger.exception(
                        "Error inferring segment %d",
                        metadata.get("segment_index", -1),
                    )
                    continue

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": caption},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            terminal = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(terminal)}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            if consumer_id is not None:
                await self._stream_manager.unsubscribe(
                    uri=rtsp_url,
                    chunk_duration=chunk_duration,
                    num_frames=num_frames,
                    consumer_id=consumer_id,
                )

    async def _infer_chat_caption(
        self,
        frames: npt.NDArray,
        request,
        prompt_text: str,
        request_id: str,
    ) -> str:
        """Run VLM inference on one segment and return the caption text."""
        full_prompt = self._build_video_prompt(prompt_text)

        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.0,
            max_tokens=(request.max_completion_tokens or request.max_tokens or 256),
        )
        engine_prompt: dict[str, Any] = {
            "prompt": full_prompt,
            "multi_modal_data": {"video": frames},
        }
        caption = ""
        async for output in self.engine_client.generate(
            engine_prompt,
            sampling_params,
            request_id,
        ):
            if output.outputs:
                caption = output.outputs[0].text
        return caption
