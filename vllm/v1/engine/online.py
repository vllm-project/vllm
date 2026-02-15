# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OnlineEngineClient: A lightweight EngineClient for GPU-less online serving.

This implements the EngineClient protocol without AsyncLLM or EngineCore,
enabling preprocessing (tokenization, rendering) and postprocessing
(detokenization) without GPU inference.
"""

from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs import DictPrompt, TokPrompt
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.input_processor import InputProcessor

logger = init_logger(__name__)


class OnlineEngineClient(EngineClient):
    """GPU-less EngineClient that only supports preprocessing/postprocessing.

    This is a Null Object at the EngineClient level, bypassing AsyncLLM
    entirely. It initializes renderer, io_processor, and input_processor
    for tokenization and rendering, but raises NotImplementedError for
    any inference-related operations.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        self.renderer = renderer = renderer_from_config(self.vllm_config)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # Convert TokPrompt --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> "OnlineEngineClient":
        """Create an OnlineEngineClient from a VllmConfig without GPU."""
        return cls(
            vllm_config=vllm_config,
        )

    # -- Task support --

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("render",)

    # -- Inference (not supported) --

    async def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | DictPrompt
        | TokPrompt
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        raise NotImplementedError(
            "OnlineEngineClient does not support inference. "
            "Use vllm serve for generation requests."
        )
        # yield is needed to make this an async generator
        yield  # type: ignore[misc] # pragma: no cover

    # -- Request management (no-op) --

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        pass

    # -- Generation control (no-op) --

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None:
        pass

    async def resume_generation(self) -> None:
        pass

    async def is_paused(self) -> bool:
        return False

    async def encode(
        self,
        prompt: PromptType | DictPrompt | TokPrompt,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError(
            "OnlineEngineClient does not support inference. "
            "Use vllm serve for encoding requests."
        )
        yield  # type: ignore[misc] # pragma: no cover

    # -- Observability (no-op / defaults) --

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self) -> None:
        pass

    async def start_profile(self) -> None:
        pass

    async def stop_profile(self) -> None:
        pass

    # -- Cache management (no-op) --

    async def reset_mm_cache(self) -> None:
        pass

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return True

    async def reset_encoder_cache(self) -> None:
        pass

    # -- Power management (no-op) --

    async def sleep(self, level: int = 1) -> None:
        pass

    async def wake_up(self, tags: list[str] | None = None) -> None:
        pass

    async def is_sleeping(self) -> bool:
        return False

    # -- LoRA (not supported) --

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    # -- Status properties --

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return RuntimeError("OnlineEngineClient does not support inference")
