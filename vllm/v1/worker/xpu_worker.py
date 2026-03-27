# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import signal
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.mem_utils import (MemorySnapshot, format_gib,
                                  memory_profiling)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.xpu_model_runner import XPUModelRunner

from .utils import request_memory

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
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU", "XPU"],
            )

    def init_device(self):
        # Ignore SIGPIPE to prevent TBX socket errors from killing the process
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        device = self.device_config.device
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            self.device = torch.device(f"xpu:{self.local_rank}")
            current_platform.set_device(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            torch.accelerator.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank
            ).total_memory
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        try:
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize distributed environment: %s. "
                "Continuing without distributed support "
                "(expected on simulators without oneCCL).",
                e,
            )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        try:
            torch.accelerator.empty_cache()
        except RuntimeError as e:
            logger.warning("torch.accelerator.empty_cache() failed: %s", e)

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        self.model_runner = XPUModelRunner(  # type: ignore
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        Overrides the GPU worker version to handle XPU simulator environments
        where device memory APIs may fail (e.g. device index -1 errors).
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Reserved %s GiB for KV cache (kv_cache_memory_bytes).",
                format_gib(kv_cache_memory_bytes),
            )
            return kv_cache_memory_bytes

        try:
            with memory_profiling(
                self.init_snapshot,
                weights_memory=int(self.model_runner.model_memory_usage),
            ) as profile_result:
                self.model_runner.profile_run()

            self.non_torch_memory = profile_result.non_torch_increase
            self.peak_activation_memory = profile_result.torch_peak_increase

            free_gpu_memory = profile_result.after_profile.free_memory

            if self.init_snapshot.free_memory >= free_gpu_memory:
                self.available_kv_cache_memory_bytes = (
                    self.requested_memory
                    - profile_result.non_kv_cache_memory
                )
            else:
                logger.warning(
                    "Memory profiling inconsistency on XPU: "
                    "init free=%s, after=%s. Using fallback.",
                    format_gib(self.init_snapshot.free_memory),
                    format_gib(free_gpu_memory),
                )
                self.available_kv_cache_memory_bytes = max(
                    self.init_snapshot.free_memory // 2, 1 * 1024 * 1024
                )

            logger.info_once(
                "Available KV cache memory: %s GiB",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )
            return int(self.available_kv_cache_memory_bytes)

        except (RuntimeError, AssertionError) as e:
            # On XPU simulators, memory profiling may fail due to device
            # index issues or unsupported memory APIs.
            logger.warning(
                "XPU memory profiling failed: %s. "
                "Running profile_run without memory tracking and using "
                "fallback memory value.",
                e,
            )
            try:
                self.model_runner.profile_run()
            except Exception as profile_err:
                logger.warning(
                    "XPU profile_run also failed: %s", profile_err
                )

            # Use a small fallback value for KV cache
            fallback_bytes = max(
                self.init_snapshot.free_memory // 2, 1 * 1024 * 1024
            )
            self.non_torch_memory = 0
            self.peak_activation_memory = 0
            self.available_kv_cache_memory_bytes = fallback_bytes
            logger.warning(
                "Using fallback KV cache memory: %s GiB",
                format_gib(fallback_bytes),
            )
            return int(fallback_bytes)
