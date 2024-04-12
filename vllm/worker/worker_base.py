import importlib
import os
from abc import ABC, abstractmethod
from typing import Dict, List

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import update_environment_variables

logger = init_logger(__name__)


class WorkerBase(ABC):
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware.
    """

    @abstractmethod
    def init_worker(self) -> None:
        """
        Constructor of the worker should only save construction arguments.
        Initialization should be done in this method, for lazy initialization.
        """
        raise NotImplementedError

    @abstractmethod
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    @abstractmethod
    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
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
    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        """Executes one model step on the given sequences."""
        raise NotImplementedError

    @abstractmethod
    def get_cache_block_size_bytes() -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
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
    """Partial implementation of WorkerBase that raises exceptions when LoRA
    methods are invoked.
    """

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def remove_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def list_loras(self) -> List[int]:
        raise ValueError(f"{type(self)} does not support LoRA")


class WorkerWrapperBase:
    """
    The whole point of this class is to lazily initialize the worker.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(self,
                 init_cached_hf_modules=False,
                 worker_module_name=None,
                 worker_class_name=None) -> None:
        self.worker_module_name = worker_module_name
        self.worker_class_name = worker_class_name
        self.init_cached_hf_modules = init_cached_hf_modules
        self.worker = None

    def update_environment_variables(self, envs: Dict[str, str]) -> None:
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, *args, **kwargs):
        if self.init_cached_hf_modules:
            from transformers.dynamic_module_utils import init_hf_modules
            init_hf_modules()
        mod = importlib.import_module(self.worker_module_name)
        worker_class = getattr(mod, self.worker_class_name)
        self.worker = worker_class(*args, **self.kwargs)

    def execute_method(self, method, *args, **kwargs):
        try:
            if hasattr(self, method):
                executor = getattr(self, method)
            else:
                executor = getattr(self.worker, method)
            return executor(*args, **kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e
