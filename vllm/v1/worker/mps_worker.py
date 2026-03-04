# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.mps_model_runner import MPSModelRunner

logger = init_logger(__name__)


class MPSWorker(Worker):
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

    def init_device(self):
        self.device = torch.device("mps")

        # Force gloo to use loopback so it skips slow network-interface
        # probing on macOS (each new_group call can take 60-70 s otherwise).
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

        # Pre-initialize torch.distributed with an in-memory HashStore.
        # Gloo TCP rendezvous hangs on macOS (even with 127.0.0.1 and
        # world_size=1).  HashStore avoids all networking.
        if not torch.distributed.is_initialized():
            store = torch.distributed.HashStore()
            torch.distributed.init_process_group(
                backend="gloo",
                store=store,
                world_size=1,
                rank=0,
            )

        # Sets up model parallelism, custom all-reduce, etc.
        # Skips init_process_group since we already did it above.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        set_random_seed(self.model_config.seed)

        # Construct the model runner on MPS device.
        self.model_runner: MPSModelRunner = MPSModelRunner(
            self.vllm_config, self.device
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is not supported on MPS, ignoring.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Sleep mode is not supported on MPS, ignoring.")

    def determine_available_memory(self) -> int:
        return self.cache_config.cpu_kvcache_space_bytes or 0

    def compile_or_warm_up_model(self) -> float:
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()
        return self.compilation_config.compilation_time
