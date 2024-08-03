from typing import (AsyncIterator, List, Mapping, Optional, Protocol,
                    runtime_checkable)

from transformers import PreTrainedTokenizer

from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.data import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput


@runtime_checkable
class AsyncEngineClient(Protocol):
    """Protocol class for Clients to AsyncLLMEngine"""

    @property
    def is_running(self) -> bool:
        ...

    @property
    def is_stopped(self) -> bool:
        ...

    @property
    def errored(self) -> bool:
        ...

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncIterator[RequestOutput]:
        """Generates outputs for a request"""

    async def encode(
        self,
        inputs: PromptInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> AsyncIterator[EmbeddingRequestOutput]:
        """Generate outputs for a request from an embedding model."""

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> PreTrainedTokenizer:
        """Get the appropriate Tokenizer for the request"""

    async def is_tracing_enabled(self) -> bool:
        pass

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        pass

    async def check_health(self) -> None:
        """Raise if unhealthy"""
