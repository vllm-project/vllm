# SPDX-License-Identifier: Apache-2.0
"""A Neuron worker class."""
import json
import os
import subprocess
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm
from torch_xla._internal.pjrt import initialize_multiprocess

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.neuron_model_runner import NeuronModelRunner

logger = init_logger(__name__)


def get_current_memory_usage(rank):

    process = subprocess.Popen("neuron-monitor",
                               shell=False,
                               stdout=subprocess.PIPE,
                               preexec_fn=os.setsid)

    try:
        outs, errs = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        outs, errs = process.communicate()
        runtime_data = json.loads(outs)['neuron_runtime_data']
        hardware_info = json.loads(outs)['neuron_hardware_info']
        if len(runtime_data) == 0:
            memory_used = 0
        else:
            memory_used = runtime_data[0]['report']['memory_used'][
                'neuron_runtime_used_bytes']['neuron_device']

        total_memory = hardware_info[
            'neuron_device_memory_size'] // hardware_info[
                'neuroncore_per_device_count']
    return memory_used, total_memory


class NeuronWorker:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        self.is_driver_worker = is_driver_worker
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.profiler = None

        if self.model_config.seed is None:
            self.model_config.seed = 0

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        self.model_runner.profile_run()
        # TODO: Find a cleaner way to handle these retries. This loop was
        # a hack to retry the invocation of neuron-monitor, which was failing
        # frequently when there were multiple workers running.
        for _ in range(10):
            try:
                memory_usage, total_memory = get_current_memory_usage(
                    self.rank)
                break
            except json.JSONDecodeError:
                continue
        kv_cache_memory_available = (total_memory *
                                     self.cache_config.gpu_memory_utilization)
        return int(kv_cache_memory_available - memory_usage)

    def init_device(self):
        if self.device_config.device.type == "cpu":

            # Initialize the distributed environment.
            init_worker_distributed_environment(self.parallel_config,
                                                self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

            self.device = xm.xla_device()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        with torch.inference_mode():
            self.model_runner: NeuronModelRunner = NeuronModelRunner(
                self.vllm_config, self.device)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        raise NotImplementedError("Profiling is not implemented.")

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self):
        self.model_runner.capture_model()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate device KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    initialize_multiprocess(rank, parallel_config.tensor_parallel_size)

    init_distributed_environment(parallel_config.world_size,
                                 rank,
                                 distributed_init_method,
                                 local_rank,
                                 backend="xla")

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
