from typing import (AsyncGenerator, List, Mapping, Optional, Protocol,
                    runtime_checkable)

from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.data import PromptType
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer


@runtime_checkable
class EngineClient(Protocol):
    """Protocol class for Clients to Engine"""

    @property
    def is_running(self) -> bool:
        ...

    @property
    def is_stopped(self) -> bool:
        ...

    @property
    def errored(self) -> bool:
        ...

    @property
    def dead_error(self) -> BaseException:
        ...

    def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """Generate outputs for a request from an embedding model."""
        ...

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        ...

    async def get_decoding_config(self) -> DecodingConfig:
        ...
        """Get the decoding configuration of the vLLM engine."""

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        ...

    async def is_tracing_enabled(self) -> bool:
        ...

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        ...

    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        ...
