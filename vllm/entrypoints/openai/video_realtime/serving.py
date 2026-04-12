# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


MIN_VIDEO_FRAMES = 2

VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><vision_end|>"

QWEN_CHAT_USER_PREFIX = "<|im_start|>user\n"
QWEN_CHAT_USER_SUFFIX = "<|im_end|>\n"
QWEN_CHAT_ASSISTANT_PREFIX = "<|im_start|>assistant\n"



class OpenAIServingRealtimeVideo(OpenAIServing):
    """Realtime video service via WebSocket streaming.

    Provides streaming video frames into StreamingInput objects with
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
    
        logger.info("OpenAIServingRealtimeVideo initialized for task: %s", self.task_type)



    async def stream_video_realtime(
            self,
            video_batch_queue: asyncio.Queue[list | None],
            _frame_idx_queue: asyncio.Queue[list | None],
            prompt_getter: Callable[[], str] | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        
        def _get_prompt() -> str:
            return prompt_getter() if prompt_getter else ""

        while True:
            batch = await video_batch_queue.get()
            if batch is None:
                # Signal for end of stream
                break
            frame_idx = await _frame_idx_queue.get()
            current_prompt = _get_prompt()
            if not batch:
                yield StreamingInput(
                    prompt=TextPrompt(prompt=current_prompt))
                continue
            num_frames = len(batch)
            frames_array = np.stack([np.array(img) for img in batch])
            if num_frames < MIN_VIDEO_FRAMES:
                repeat = (MIN_VIDEO_FRAMES + num_frames - 1) // num_frames
                frames_array = np.tile(frames_array, (repeat, 1, 1, 1))[:MIN_VIDEO_FRAMES]
                num_frames = MIN_VIDEO_FRAMES
            height, width = frames_array.shape[1], frames_array.shape[2]
            fps = 1.0
            meta_data = {
                "total_num_frames": num_frames,
                "fps": fps,
                "duration": num_frames / fps,
                "video_backend": "realtime_video",
                "frame_indices": frame_idx, 
                "do_sample_frames": True,
                "width": width,
                "height": height,
            }
            if VIDEO_PLACEHOLDER not in current_prompt:
                user_content = current_prompt + " " + VIDEO_PLACEHOLDER
            else:
                user_content = current_prompt
            effective_prompt = (
                QWEN_CHAT_USER_PREFIX + user_content + QWEN_CHAT_USER_SUFFIX +
                QWEN_CHAT_ASSISTANT_PREFIX
            )
            has_text = current_prompt != ""
            prompt = TextPrompt(prompt=effective_prompt, 
                                multi_modal_data={"video": (frames_array, meta_data)},
                                has_text=has_text)
            yield StreamingInput(prompt=prompt)
            