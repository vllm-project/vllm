"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Tuple, Set, Optional

import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.lora.request import LoRARequest
from vllm.utils import is_xpu
from vllm.device_utils import (device_empty_cache, device_synchronize,
                               mem_get_info, get_distribute_backend)

from vllm.logger import init_logger

logger = init_logger(__name__) 

class Worker(Worker):
    """A worker class that executes (a partition of) the model on a GPU.

    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        assert device_config.device_type == "xpu"
        model_config = Worker._verify_and_get_model_config(model_config)
        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
        )         

    @staticmethod
    def _verify_and_get_model_config(config: ModelConfig) -> ModelConfig:
        if (config.enforce_eager == False):
            logger.warning(f"CUDA graph is not supported on XPU, fallback to the eager mode.")
            config.enforce_eager = True
        return config
    
    def init_model(self, cupy_port: Optional[int] = None) -> None:
        if self.device_config.device.type == "xpu" and is_xpu():
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.xpu.set_device(self.device)
            torch.xpu.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(self.local_rank).total_memory
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        Worker.init_distributed_environment(
            self.parallel_config,
            self.rank,
            self.distributed_init_method,
        )
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
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
        total_gpu_memory = torch.xpu.get_device_properties(self.local_rank).total_memory
        print(f"rank:{self.local_rank}, used_memory:{used_memory}")
        
        free_gpu_memory = total_gpu_memory - used_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory

        cache_block_size = self.get_cache_block_size_bytes(
            block_size, cache_dtype)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.xpu.empty_cache()
        print(f"num_gpu_blocks:{num_gpu_blocks}, num_cpu_blocks:{num_cpu_blocks}")
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config,
                                        self.device_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)


    # def cache_swap(
    #     self,
    #     blocks_to_swap_in: Dict[int, int],
    #     blocks_to_swap_out: Dict[int, int],
    #     blocks_to_copy: Dict[int, List[int]],
    # ) -> None:
    #     # Issue cache operations.
    #     issued_cache_op = False
    #     if blocks_to_swap_in:
    #         self.cache_engine.swap_in(blocks_to_swap_in)
    #         issued_cache_op = True
    #     if blocks_to_swap_out:
    #         self.cache_engine.swap_out(blocks_to_swap_out)
    #         issued_cache_op = True
    #     if blocks_to_copy:
    #         self.cache_engine.copy(blocks_to_copy)
    #         issued_cache_op = True

    #     cache_events = self.cache_events if issued_cache_op else None

    #     # Wait for cache operations to finish.
    #     # TODO(woosuk): Profile swapping overhead and optimize if needed.
    #     if cache_events is not None:
    #         for event in cache_events:
    #             event.wait()

    # @torch.inference_mode()
    # def execute_model(
    #     self,
    #     seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
    #     blocks_to_swap_in: Optional[Dict[int, int]] = None,
    #     blocks_to_swap_out: Optional[Dict[int, int]] = None,
    #     blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    # ) -> Optional[SamplerOutput]:
    #     if self.is_driver_worker:
    #         assert seq_group_metadata_list is not None
    #         num_seq_groups = len(seq_group_metadata_list)
    #         assert blocks_to_swap_in is not None
    #         assert blocks_to_swap_out is not None
    #         assert blocks_to_copy is not None
    #         data = {
    #             "num_seq_groups": num_seq_groups,
    #             "blocks_to_swap_in": blocks_to_swap_in,
    #             "blocks_to_swap_out": blocks_to_swap_out,
    #             "blocks_to_copy": blocks_to_copy,
    #         }
    #         broadcast_tensor_dict(data, src=0)
    #     else:
    #         data = broadcast_tensor_dict(src=0)
    #         num_seq_groups = data["num_seq_groups"]
    #         blocks_to_swap_in = data["blocks_to_swap_in"]
    #         blocks_to_swap_out = data["blocks_to_swap_out"]
    #         blocks_to_copy = data["blocks_to_copy"]

    #     self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

    #     # If there is no input, we don't need to execute the model.
    #     if num_seq_groups == 0:
    #         return {}

    #     output = self.model_runner.execute_model(seq_group_metadata_list,
    #                                              self.gpu_cache)
    #     return output

    # def get_cache_block_size_bytes(self, block_size: int,
    #                                cache_dtype: str) -> int:
    #     """Get the size of the KV cache block size in bytes.
    #     """
    #     return CacheEngine.get_cache_block_size(block_size, cache_dtype,
    #                                             self.model_config,
    #                                             self.parallel_config)

    @staticmethod
    def init_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        """Initialize the distributed environment."""
        
        def all_reduce_warmup():
            # if torch.xpu.is_available():
            #     torch.distributed.all_reduce(torch.zeros(1).xpu())
            pass
        
        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch world "
                    "size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {parallel_config.world_size}).")
        elif not distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            # backend = (device_config)
            torch.distributed.init_process_group(
                backend="ccl",
                world_size=parallel_config.world_size,
                rank=rank,
                init_method=distributed_init_method,
            )

        # A small all_reduce for warmup.
        all_reduce_warmup()

        ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                          parallel_config.pipeline_parallel_size)






