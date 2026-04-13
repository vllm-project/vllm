# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for streaming video input via WebSocket."""

import asyncio
from collections.abc import AsyncGenerator, Callable

import numpy as np

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import StreamingInput, TextPrompt
from vllm.logger import init_logger

logger = init_logger(__name__)

# Default text prompt when client does not send one (video-only input).
# we utilize it as the default summarize text
DEFAULT_VIDEO_PROMPT = ""

# Qwen2-VL/Qwen3-VL require at least 2 frames; different models have different frame
# requirements. Pad by repeating frames when insufficient, to satisfy model constraints.
MIN_VIDEO_FRAMES = 2

# Video placeholder required by the model so prompt replacement can inject video.
# Used by Qwen2-VL/Qwen3-VL; other models may use different placeholders (set
# prompt via session.update including the correct placeholder if needed).
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

# Qwen-style chat wrapper so the model generates a full assistant reply instead of
# stopping after one token (EOS). Without this, raw prompt has no "assistant" turn
# and the model may emit EOS immediately.
QWEN_CHAT_USER_PREFIX = "<|im_start|>user\n"
QWEN_CHAT_USER_SUFFIX = "<|im_end|>\n"
QWEN_CHAT_ASSISTANT_PREFIX = "<|im_start|>assistant\n"


class OpenAIServingRealtimeVideo(OpenAIServing):
    """Realtime video understanding via WebSocket streaming.

    Transforms streamed video frames into StreamingInput objects with
    multi_modal_data for engine.generate().
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        log_error_stack: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )
        self.task_type = "realtime_video"
        logger.info(
            "OpenAIServingRealtimeVideo initialized for task: %s", self.task_type
        )

    async def stream_video_realtime(
        self,
        video_batch_queue: asyncio.Queue[list | None],
        frame_idx_queue: asyncio.Queue[list | None],
        prompt_text: str = DEFAULT_VIDEO_PROMPT,
        prompt_getter: Callable[[], str] | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Turn queued video batches into StreamingInput for engine.generate().
        """

        def _get_prompt() -> str:
            user_prompt = prompt_getter() if prompt_getter else prompt_text
            return user_prompt if prompt_getter else ""

        while True:
            batch = await video_batch_queue.get()
            frame_idx = await frame_idx_queue.get()
            if batch is None:
                break
            current_prompt = _get_prompt()
            if not batch:
                yield StreamingInput(prompt=TextPrompt(prompt=current_prompt))
                continue
            num_frames = len(batch)
            frames_array = np.stack([np.array(img) for img in batch])
            if num_frames < MIN_VIDEO_FRAMES:
                repeat = (MIN_VIDEO_FRAMES + num_frames - 1) // num_frames
                frames_array = np.tile(frames_array, (repeat, 1, 1, 1))[:MIN_VIDEO_FRAMES]
                num_frames = MIN_VIDEO_FRAMES
            height, width = frames_array.shape[1], frames_array.shape[2]
            fps = 1.0
            metadata = {
                "total_num_frames": num_frames,
                "fps": fps, # this also needs to be sent in
                "duration": num_frames / fps,
                "video_backend": "realtime_stream",
                "frames_indices": frame_idx,
                "do_sample_frames": True,
                "width": width,
                "height": height,
            }
            if VIDEO_PLACEHOLDER not in current_prompt:
                user_content = current_prompt + " " + VIDEO_PLACEHOLDER
            else:
                user_content = current_prompt
            effective_prompt = (
                QWEN_CHAT_USER_PREFIX
                + user_content
                + QWEN_CHAT_USER_SUFFIX
                + QWEN_CHAT_ASSISTANT_PREFIX
            )
            has_text = current_prompt != ""
            prompt = TextPrompt(prompt=effective_prompt, 
                                multi_modal_data={"video": (frames_array, metadata)},
                                has_text=has_text)
            yield StreamingInput(prompt=prompt)