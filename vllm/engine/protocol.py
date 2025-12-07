# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from vllm.config import ModelConfig, RendererConfig, VllmConfig
from vllm.inputs.data import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    vllm_config: VllmConfig
    renderer_config: RendererConfig
    model_config: ModelConfig
    input_processor: InputProcessor
    io_processor: IOProcessor | None

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @property
    @abstractmethod
    def errored(self) -> bool: ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
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
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    @abstractmethod
    async def get_tokenizer(self) -> TokenizerLike:
        """Get the tokenizer"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    @abstractmethod
    async def do_log_stats(self) -> None: ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    @abstractmethod
    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight requests
                to finish before pausing. When ``False`` (default), aborts in-flight
                requests immediately.
            clear_cache: Whether to clear KV and prefix caches after draining.
        """
        ...

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError
