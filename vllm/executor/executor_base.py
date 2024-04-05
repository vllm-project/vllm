from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VisionLanguageConfig)
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on a specific device
    type (e.g., CPU, GPU, Neuron, etc.). Or it can be a distributed executor
    that can execute the model on multiple devices.
    """

    @abstractmethod
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        speculative_config: Optional[SpeculativeConfig],
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def get_max_allowed_kv_blocks(self) -> tuple[int, int]:
        """Profile the model on-device to determine the maximum number of KV
        blocks that can be allocated.

        Returns a tuple[num_device_blocks, num_cpu_blocks], where
            num_device_blocks refers to the number of blocks in the "active" KV
            cache (e.g. where blocks are appended to), and num_cpu_blocks refers
            to the number of blocks in the "passive" KV cache (e.g. where blocks
            are swapped to).

        Examples:
            - The GPUExecutor will return [num_gpu_blocks, num_cpu_blocks].
            - A future CPUExecutor can return [num_cpu_blocks, 0] or
                [num_cpu_blocks, num_swap_cpu_blocks].
        """
        raise NotImplementedError


    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Given a fully-specified cache config, initialize the KV cache. This
        is separate from init_workers as profiling may be required to determine
        the maxmimum allowed KV cache size.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_loras(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError


class ExecutorAsyncBase(ExecutorBase):

    @abstractmethod
    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    @abstractmethod
    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError
