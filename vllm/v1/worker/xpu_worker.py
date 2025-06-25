# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import (Worker,
                                       init_worker_distributed_environment)
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
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def init_device(self):
        if self.device_config.device.type == "xpu" and current_platform.is_xpu(
        ):
            self.device = torch.device(f"xpu:{self.local_rank}")
            current_platform.set_device(self.device)
            current_platform.empty_cache()
            self.init_gpu_memory = current_platform.get_device_total_memory(
                self.local_rank)
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        ENV_CCL_ZE_IPC_EXCHANGE = os.getenv("CCL_ZE_IPC_EXCHANGE", "pidfd")
        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv("LOCAL_WORLD_SIZE",
                                         str(self.parallel_config.world_size))
        os.environ["CCL_ZE_IPC_EXCHANGE"] = ENV_CCL_ZE_IPC_EXCHANGE
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)

        # global all_reduce needed for overall oneccl warm up
        torch.distributed.all_reduce(torch.zeros(1).xpu(),
                                     group=get_world_group().device_group)

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner = XPUModelRunner(  # type: ignore
            self.vllm_config, self.device)
