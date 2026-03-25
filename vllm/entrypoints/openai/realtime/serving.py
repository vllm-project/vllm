# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Literal, cast

import numpy as np

from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsRealtime,
    SupportsRealtimeVideo,
)
from vllm.renderers.inputs.preprocess import parse_model_prompt

logger = init_logger(__name__)


class OpenAIServingRealtime(OpenAIServing):
    """Realtime audio transcription service via WebSocket streaming.

    Provides streaming audio-to-text transcription by transforming audio chunks
    into StreamingInput objects that can be consumed by the engine.
    """

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

        self.task_type: Literal["realtime"] = "realtime"

        logger.info("OpenAIServingRealtime initialized for task: %s", self.task_type)

    @cached_property
    def model_cls(self) -> type[SupportsRealtime]:
        """Get the model class that supports transcription."""
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsRealtime], model_cls)

    @cached_property
    def video_model_cls(self) -> type[SupportsRealtimeVideo]:
        """Get the model class that supports realtime video."""
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsRealtimeVideo], model_cls)

    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform audio stream into StreamingInput for engine.generate().

        Args:
            audio_stream: Async generator yielding float32 numpy audio arrays
            input_stream: Queue containing context token IDs from previous
                generation outputs. Used for autoregressive multi-turn
                processing where each generation's output becomes the context
                for the next iteration.

        Yields:
            StreamingInput objects containing audio prompts for the engine
        """
        model_config = self.model_config
        renderer = self.renderer

        # mypy is being stupid
        # TODO(Patrick) - fix this
        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.model_cls.buffer_realtime_audio(
                audio_stream, input_stream, model_config
            ),
        )

        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])

            yield StreamingInput(prompt=engine_input)

    async def understand_video_realtime(
        self,
        video_stream: AsyncGenerator[np.ndarray, None],
        query: str | None,
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform video stream into StreamingInput for engine.generate().

        Args:
            video_stream: Async generator yielding uint8 numpy frames (H,W,3)
            query: Optional text query about the video content
            input_stream: Queue containing context token IDs from previous
                generation outputs for autoregressive multi-turn processing.

        Yields:
            StreamingInput objects containing video prompts for the engine
        """
        model_config = self.model_config
        renderer = self.renderer

        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.video_model_cls.buffer_realtime_video(
                video_stream, query, input_stream, model_config
            ),
        )

        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])

            yield StreamingInput(prompt=engine_input)
