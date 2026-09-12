# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections import deque
from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Literal, cast

import numpy as np

from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.renderers.params import TokenizeParams

from .protocol import RealtimeSessionConfig

logger = init_logger(__name__)


class OpenAIServingRealtime(GenerateBaseServing):
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

    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        session_config: RealtimeSessionConfig | None = None,
        prefix_texts: deque[str] | None = None,
        request_id: str | None = None,
        segment_traces: deque[dict] | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform audio stream into StreamingInput for engine.generate().

        Args:
            audio_stream: Async generator yielding float32 numpy audio arrays
            input_stream: Queue containing context token IDs from previous
                generation outputs. Used for autoregressive multi-turn
                processing where each generation's output becomes the context
                for the next iteration.
            session_config: Optional session-level configuration (language,
                prompt, segment duration) from the client's session.update.
            prefix_texts: Deque shared with _run_generation carrying the
                actual prefix string for each segment so deltas can be
                computed by reconstructing the full raw_decoded.

        Yields:
            StreamingInput objects containing audio prompts for the engine
        """
        model_config = self.model_config
        renderer = self.renderer

        kwargs: dict = {}
        if session_config is not None:
            kwargs["segment_duration_s"] = session_config.segment_duration_s
            if session_config.language is not None:
                kwargs["language"] = session_config.language
            if session_config.prompt is not None:
                kwargs["prompt"] = session_config.prompt
            if session_config.rollback_tokens is not None:
                kwargs["rollback_tokens"] = session_config.rollback_tokens
            if session_config.unfixed_chunks is not None:
                kwargs["unfixed_chunks"] = session_config.unfixed_chunks
            if session_config.max_prefix_tokens is not None:
                kwargs["max_prefix_tokens"] = session_config.max_prefix_tokens
            if session_config.max_audio_s is not None:
                kwargs["max_audio_s"] = session_config.max_audio_s

        # mypy is being stupid
        # TODO(Patrick) - fix this
        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.model_cls.buffer_realtime_audio(
                audio_stream,
                input_stream,
                model_config,
                prefix_texts=prefix_texts,
                request_id=request_id,
                segment_traces=segment_traces,
                **kwargs,
            ),
        )

        max_tokens = (
            session_config.realtime_max_tokens
            if session_config is not None
            and session_config.realtime_max_tokens is not None
            else self.model_cls.realtime_max_tokens
        )
        tok_params = TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=max_tokens,
            add_special_tokens=False,
        )

        segment_id = 0
        async for prompt in stream_input_iter:
            segment_id += 1
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_input,) = await renderer.render_cmpl_async(
                [parsed_prompt], tok_params
            )

            prompt_tokens = len(engine_input.get("prompt_token_ids") or ())
            if prompt_tokens + max_tokens > model_config.max_model_len:
                raise ValueError(
                    "QWEN3_ASR_REALTIME_CONTEXT_LIMIT: "
                    f"segment_id={segment_id}, prompt_tokens={prompt_tokens}, "
                    f"requested_output_tokens={max_tokens}, "
                    f"max_model_len={model_config.max_model_len}. "
                    "The full audio + full text prefix is preserved; increase "
                    "--max-model-len or start a new realtime session."
                )

            aut_stable_window_mode = os.environ.get(
                "QWEN3_ASR_REALTIME_MODE", ""
            ).strip().lower() == "aut_stable_window_kv"
            segment_trace = (
                segment_traces[-1]
                if aut_stable_window_mode and segment_traces
                else None
            )
            yield StreamingInput(
                prompt=engine_input,
                segment_id=segment_id,
                new_audio_feature_count=(1 if aut_stable_window_mode else 0),
                final_segment=(
                    bool(segment_trace.get("aut_window_commit", False))
                    if segment_trace is not None
                    else False
                ),
            )
