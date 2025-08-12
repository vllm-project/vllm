from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ObservabilityConfig,
                         ParallelConfig, PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on a specific device
    type (e.g., CPU, GPU, Neuron, etc.). Or it can be a distributed executor
    that can execute the model on multiple devices.
    """

    uses_ray: bool  # whether the executor uses Ray for orchestration.

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        observability_config: Optional[ObservabilityConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.multimodal_config = multimodal_config
        self.speculative_config = speculative_config
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        pass

    @abstractmethod
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        Normally, this should simply delegate to the underlying Worker. Some
        ExecutorBase may require modification of the result, e.g. to ensure the
        selected cache sizes are compatible with all workers.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences."""
        raise NotImplementedError

    def stop_remote_worker_execution_loop(self) -> None:
        """Releases parallel workers from model loop."""
        return

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError  # type: ignore

    @abstractmethod
    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @abstractmethod
    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError  # type: ignore

    @abstractmethod
    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        return

    def __del__(self):
        self.shutdown()


class ExecutorAsyncBase(ExecutorBase):

    @abstractmethod
    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()
