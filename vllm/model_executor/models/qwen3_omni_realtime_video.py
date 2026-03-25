# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni realtime video streaming support.

This module provides a thin subclass of the Qwen3-Omni thinker that
implements :class:`SupportsRealtimeVideo`. It buffers incoming video
frames from a WebSocket stream and yields prompts that the engine can
process incrementally.

A separate model class is required because the multimodal processor
registry binds one processor per model class. The realtime video
endpoint needs a different processor configuration (no caching, dynamic
frame counts) than the standard chat completion endpoint.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import ClassVar, Literal

import numpy as np

from vllm.config import ModelConfig
from vllm.inputs import PromptType, TokensPrompt
from vllm.model_executor.models.interfaces import SupportsRealtimeVideo
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

_DEFAULT_QUERY = "Describe what you see in the video."

# Maximum number of frames to buffer before yielding a prompt.
# This prevents unbounded memory growth if the client sends many
# frames before committing.
_MAX_BUFFER_FRAMES = 64


class Qwen3OmniRealtimeVideoBuffer:
    """Buffer that accumulates video frames for batch processing.

    Frames are collected until either the stream ends or
    ``max_frames`` is reached, at which point a batch is yielded.
    """

    def __init__(self, max_frames: int = _MAX_BUFFER_FRAMES):
        self._max_frames = max_frames
        self._frames: list[np.ndarray] = []

    def write_frame(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def is_full(self) -> bool:
        return len(self._frames) >= self._max_frames

    def read_batch(self) -> np.ndarray | None:
        """Return buffered frames as (N, H, W, 3) array and clear."""
        if not self._frames:
            return None
        batch = np.stack(self._frames, axis=0)
        self._frames.clear()
        return batch

    @property
    def frame_count(self) -> int:
        return len(self._frames)


# NOTE: A separate model class is required here because the multimodal
# processor registry binds one processor per model class. The realtime
# video endpoint needs a different processor than the standard chat
# completion endpoint, so we register it on this subclass.
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniRealtimeVideoGeneration(
    Qwen3OmniMoeThinkerForConditionalGeneration,
    SupportsRealtimeVideo,
):
    """Qwen3-Omni with streaming video input via the realtime API."""

    supports_realtime_video: ClassVar[Literal[True]] = True
    realtime_video_max_tokens: ClassVar[int] = 2048

    @classmethod
    async def buffer_realtime_video(
        cls,
        video_stream: AsyncGenerator[np.ndarray, None],
        query: str | None,
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        """Buffer video frames and yield prompts for the engine.

        The method collects frames from *video_stream* (each an
        ``(H, W, 3)`` uint8 numpy array) and batches them into a
        single ``(N, H, W, 3)`` array. When the stream ends or the
        buffer is full a prompt is yielded containing the frames as
        ``multi_modal_data["video"]`` together with the Qwen3-Omni
        chat template.

        Args:
            video_stream: Async generator yielding RGB frames.
            query: Optional user question about the video. Falls back
                to a generic "describe" prompt if *None*.
            input_stream: Token ID queue for autoregressive context
                (fed back from previous generation outputs).
            model_config: The engine's model configuration.
        """
        tokenizer = cached_tokenizer_from_config(model_config)
        query_text = query or _DEFAULT_QUERY

        # Build the Qwen3-Omni chat prompt with video placeholder.
        video_placeholder = cls.get_placeholder_str("video", 0)
        prompt_text = (
            f"<|im_start|>system\n{_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{video_placeholder}"
            f"{query_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_token_ids = tokenizer.encode(prompt_text)

        buffer = Qwen3OmniRealtimeVideoBuffer(max_frames=_MAX_BUFFER_FRAMES)

        async for frame in video_stream:
            buffer.write_frame(frame)

            # Yield a batch when the buffer is full (back-pressure).
            if buffer.is_full():
                frames = buffer.read_batch()
                if frames is not None:
                    yield TokensPrompt(
                        prompt_token_ids=prompt_token_ids,
                        multi_modal_data={"video": frames},
                    )

        # Flush any remaining frames.
        remaining = buffer.read_batch()
        if remaining is not None:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"video": remaining},
            )
