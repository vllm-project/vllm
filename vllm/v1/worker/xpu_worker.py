# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import torch
import torch.distributed

import vllm.envs as envs
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

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def compile_or_warm_up_model(self) -> None:
        pass

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
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        torch.xpu.empty_cache()

        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

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
