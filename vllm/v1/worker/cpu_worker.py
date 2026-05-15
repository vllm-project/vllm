# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
import sys
from typing import Any

import psutil
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.cpu_resource_utils import (
    get_allowed_cpu_list,
    get_memory_node_info,
    get_visible_memory_node,
)
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


class CPUWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # TODO: use numactl for process setup
        # TODO: optimize for `interleaved` policy
        # Bind memory node
        allowed_memory_nodes = get_visible_memory_node()
        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        # TODO: some CI hosts are not correctly set, change to assertion
        # after fix
        if cpu_core.numa_node not in allowed_memory_nodes:
            logger.warning(
                "Node %s is not in available memory nodes %s.",
                cpu_core.numa_node,
                allowed_memory_nodes,
            )

        torch.ops._C.init_cpu_memory_env([cpu_core.numa_node])

        memory_status = get_memory_node_info(cpu_core.numa_node)
        memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self.requested_cpu_memory = math.ceil(
            memory_status.total_memory * memory_fraction
        )
        available_memory = memory_status.available_memory

        if (
            vllm_config.cache_config.kv_cache_memory_bytes is None
            and self.requested_cpu_memory > available_memory
        ):
            raise ValueError(
                f"Available memory on node {cpu_core.numa_node} "
                f"({format_gib(available_memory)}/"
                f"{format_gib(memory_status.total_memory)} GiB) on startup "
                f"is less than desired CPU memory utilization "
                f"({vllm_config.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_cpu_memory)} GiB). "
                "On the CPU backend, the `--gpu-memory-utilization` flag "
                "controls the fraction of CPU memory reserved (despite its "
                "name). To resolve: decrease `--gpu-memory-utilization` "
                "(e.g. `--gpu-memory-utilization 0.5`) "
                "or reduce CPU memory used by other processes."
            )

        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.parallel_config.disable_custom_all_reduce = True

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )

    def init_device(self):
        # Check whether critical libraries are loaded
        def check_preloaded_libs(name: str):
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            if name not in ld_preload_list:
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )

        if sys.platform.startswith("linux"):
            check_preloaded_libs("libtcmalloc")
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                check_preloaded_libs("libiomp")

        def skip_set_num_threads(x: int):
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        torch.set_num_threads = skip_set_num_threads

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

        # Construct the model runner
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu")
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        self.model_runner.warming_up_model()

        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        memory_status = get_memory_node_info(cpu_core.numa_node)
        available_memory = memory_status.available_memory
        explicit_kv_cache_size = self.cache_config.kv_cache_memory_bytes

        kv_cache_size = None
        msg = None
        if explicit_kv_cache_size is not None:
            if explicit_kv_cache_size > available_memory:
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(explicit_kv_cache_size)} GiB). "
                    "Decrease --kv-cache-memory-bytes, VLLM_CPU_KVCACHE_SPACE, "
                    "or reduce CPU memory used by other processes."
                )
            kv_cache_size = explicit_kv_cache_size
            msg = (
                f"Explicitly set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}."
            )
        else:
            consumed_memory = psutil.Process(os.getpid()).memory_info().rss
            requested_memory_for_kv = int(self.requested_cpu_memory - consumed_memory)
            if (
                requested_memory_for_kv <= 0
                or requested_memory_for_kv > available_memory
            ):
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(requested_memory_for_kv)}/"
                    f"{format_gib(self.requested_cpu_memory)} GiB). "
                    "Reduce CPU memory used by other processes."
                )
            kv_cache_size = requested_memory_for_kv
            msg = (
                f"Auto set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}, with "
                f"{format_gib(self.requested_cpu_memory)} GiB requested memory"
                f" for the worker. {format_gib(consumed_memory)} GiB"
                f" memory was consumed by non-kv usages."
            )

        logger.info(msg)

        return kv_cache_size

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        # Note: the model has been compiled in determine_available_memory()
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
