# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Must be imported firstly
import vllm.v1.worker.cpu.shm  # noqa # isort: skip

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib, get_cpu_memory
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.simulated_cpu_model_runner import SimulatedCPUModelRunner
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


class SimulatedCPUWorker(Worker):
    """CPU worker for simulated forward and virtual KV-cache simulation."""

    def init_device(self) -> None:
        self.device = torch.device("cpu")
        self.parallel_config.disable_custom_all_reduce = True

        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)
        self.model_runner = SimulatedCPUModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        simulated_kv_cache_size = self.cache_config.kv_cache_memory_bytes
        if simulated_kv_cache_size is None:
            simulated_kv_cache_size = int(
                get_cpu_memory() * self.cache_config.gpu_memory_utilization
            )
        logger.info(
            "Using %s GiB simulated KV cache memory for virtual KV cache.",
            format_gib(simulated_kv_cache_size),
        )
        return simulated_kv_cache_size

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported for simulated CPU, ignore it.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported for simulated CPU, ignore it.")
