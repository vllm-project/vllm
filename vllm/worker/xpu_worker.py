"""A XPU worker class."""
import gc
import os
from typing import List, Optional, Tuple

import intel_extension_for_pytorch  # noqa: F401
import oneccl_bindings_for_pytorch  # noqa: F401
import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import is_xpu
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase
from vllm.worker.xpu_model_runner import XPUModelRunner

logger = init_logger(__name__)


class XPUWorker(LoraNotSupportedWorkerBase, Worker):
    """A worker class that executes (a partition of) the model on a GPU.
    
    Each worker is associated with a single XPU device. The worker is 
    responsible for maintaining the KV cache and executing the model on the 
    XPU. In case of distributed inference, each worker is assigned a partition
    of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        assert device_config.device_type == "xpu"
        assert is_xpu()

        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.prompt_adapter_config = prompt_adapter_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.multimodal_config = multimodal_config

        self.model_runner = XPUModelRunner(  # type: ignore
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=self.load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            multimodal_config=multimodal_config,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        self.gpu_cache: Optional[List[List[torch.Tensor]]]

    def init_device(self) -> None:
        if self.device_config.device.type == "xpu" and is_xpu():
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.xpu.set_device(self.device)
            torch.xpu.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank).total_memory
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        self.init_worker_distributed_environment()
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    # keep this method for `empty_cache` and `synchronize` api
    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.xpu.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.xpu.synchronize()
        used_memory = torch.xpu.memory_allocated()
        total_gpu_memory = torch.xpu.get_device_properties(
            self.local_rank).total_memory
        free_gpu_memory = total_gpu_memory - used_memory

        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        gc.collect()
        torch.xpu.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def _warm_up_model(self) -> None:
        # IPEX don't support capture graph yet
        pass

    def init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method

        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch "
                    "world size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {parallel_config.world_size}).")
        elif not distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            # use sockets as default Level zero IPC exchange backend. By
            # default oneccl will use `drmfd` as mechanism which need extra
            # dependency (libdrm and drm headers) on your system.
            ENV_CCL_ZE_IPC_EXCHANGE = os.getenv("CCL_ZE_IPC_EXCHANGE",
                                                "sockets")
            os.environ['CCL_ZE_IPC_EXCHANGE'] = ENV_CCL_ZE_IPC_EXCHANGE
            init_distributed_environment(
                world_size=parallel_config.world_size,
                rank=rank,
                distributed_init_method=distributed_init_method,
                local_rank=self.local_rank,
                backend="ccl")

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size)
