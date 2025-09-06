# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A Neuron worker class."""
from typing import Optional

import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)


class NeuronWorker(WorkerBase):
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False) -> None:
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.device = self.device_config.device
        self.model_runner = self.get_neuronx_distributed_model_runner(
            vllm_config, self.device)

    def init_device(self) -> None:
        self.init_distributed_environment()

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def determine_available_memory(self):
        # Note: This is not needed for Neuron, thus setting to 1GB as a
        # placeholder. This will be implemented in the navtive integration
        # phase
        return 1024 * 1024 * 1024  # 1GB

    def execute_model(
            self, scheduler_output: "SchedulerOutput"
    ) -> Optional[ModelRunnerOutput]:
        assert self.model_runner is not None
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        raise NotImplementedError

    def get_neuronx_distributed_model_runner(self, vllm_config, device):
        from vllm.v1.worker.neuronx_distributed_model_runner import (
            NeuronxDistributedModelRunner)
        return NeuronxDistributedModelRunner(vllm_config=vllm_config,
                                             device=device)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def load_model(self):
        assert self.model_runner is not None
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # Skip for NeuronX Distributed Inference
        return None

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        assert self.model_runner is not None
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        assert self.model_runner is not None
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def init_distributed_environment(self):
        """
        vLLM still needs the environment initialized when TP/PP > 1
        """
        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        ensure_model_parallel_initialized(
            1,
            1,
        )
