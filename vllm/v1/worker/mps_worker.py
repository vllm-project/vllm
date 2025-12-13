# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS (Metal Performance Shaders) worker for Apple Silicon."""

import os
from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.mps_model_runner import MPSModelRunner

logger = init_logger(__name__)


class MPSWorker(Worker):
    """Worker for MPS (Metal Performance Shaders) on Apple Silicon.

    Uses MPS GPU acceleration for model inference on Apple Silicon.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.parallel_config.disable_custom_all_reduce = True

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        self.profiler: Any | None = None
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, worker_name=worker_name, use_gzip=False
                ),
            )
        else:
            self.profiler = None

    def init_device(self):
        """Initialize the device for MPS platform.

        Uses MPS GPU acceleration for model inference.
        """
        # Use MPS device for GPU acceleration
        self.device = torch.device("mps")

        # Enable MPS fallback for unsupported ops
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]

        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner using MPS model runner
        self.model_runner: MPSModelRunner = MPSModelRunner(
            self.vllm_config, self.device
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on MPS, ignoring.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on MPS, ignoring.")
        pass

    def determine_available_memory(self) -> int:
        """Return available memory for KV cache."""
        return self.cache_config.cpu_kvcache_space_bytes or 0

    def compile_or_warm_up_model(self) -> None:
        """Warm up the model."""
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            if self.local_rank == 0:
                logger.info(
                    self.profiler.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=50
                    )
                )
