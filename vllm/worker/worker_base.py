from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata


class WorkerBase(ABC):
    @abstractmethod
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
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


class LoraNotSupportedWorkerBase(WorkerBase):
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def remove_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def list_loras(self) -> List[int]:
        raise ValueError(f"{type(self)} does not support LoRA")
