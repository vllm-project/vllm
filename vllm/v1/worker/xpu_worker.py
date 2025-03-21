# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import torch
import torch.distributed

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.xpu_model_runner import XPUModelRunner

logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method, is_driver_worker)
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()

    # we provide this function due to `torch.xpu.mem_get_info()` doesn't
    # return correct free_gpu_memory on intel client GPU. We need to
    # calculate/estiamte it.
    def xpu_get_mem_info(self):
        if current_platform.is_data_center_gpu():
            return torch.xpu.mem_get_info()
        else:
            _, total_gpu_memory = torch.xpu.mem_get_info()
            # FIXME: memory_allocated() doesn't count non-torch allocations,
            # and we don't have any API to get it. so we mark it as 128MB.
            used_memory = torch.xpu.memory_allocated()
            non_torch_allocations = 128 * 1024 * 1024
            free_gpu_memory = total_gpu_memory - (used_memory +
                                                  non_torch_allocations)
            return free_gpu_memory, total_gpu_memory

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
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
        torch.xpu.reset_peak_memory_stats()

        free_gpu_memory, total_gpu_memory = torch.xpu.mem_get_info()
        current_allocated_bytes = torch.xpu.memory_allocated()
        msg = ("Before memory profiling run, "
               f"total GPU memory: {total_gpu_memory / 1024**2:.2f} MB, "
               f"model load takes {current_allocated_bytes / 1024**2:.2f} MB, "
               f"free gpu memory is {free_gpu_memory / 1024**2:.2f} MB.")
        logger.info(msg)
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        free_gpu_memory, _ = self.xpu_get_mem_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_gpu_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.xpu.memory_stats()["allocated_bytes.all.peak"]

        torch.xpu.empty_cache()
        torch_allocated_bytes = torch.xpu.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = self.xpu_get_mem_info(
        )[1] - self.xpu_get_mem_info()[0]

        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        msg = ("After memory profiling run, "
               f"peak memory usage is {peak_memory / 1024**2:.2f} MB,"
               f"torch mem is {torch_allocated_bytes / 1024**2:.2f} MB, "
               f"non-torch mem is {non_torch_allocations / 1024**2:.2f} MB, "
               f"free gpu memory is {free_gpu_memory / 1024**2:.2f} MB.")
        logger.info(msg)

        return int(available_kv_cache_memory)

    def init_device(self):
        if self.device_config.device.type == "xpu" and current_platform.is_xpu(
        ):
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.xpu.set_device(self.device)
            torch.xpu.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank).total_memory
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        self.model_runner = XPUModelRunner(  # type: ignore
            self.vllm_config, self.device)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""

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
        # oneapi 2025 will use pidfd as default
        ENV_CCL_ZE_IPC_EXCHANGE = os.getenv("CCL_ZE_IPC_EXCHANGE", "drmfd")
        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv("LOCAL_WORLD_SIZE",
                                         str(parallel_config.world_size))
        os.environ["CCL_ZE_IPC_EXCHANGE"] = ENV_CCL_ZE_IPC_EXCHANGE
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(local_rank)
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=local_rank,
            backend="ccl")

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
    # global all_reduce needed for overall oneccl warm up
    torch.distributed.all_reduce(torch.zeros(1).xpu())
