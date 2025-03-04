# SPDX-License-Identifier: Apache-2.0
"""A TPU worker class."""
import os
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.tpu_model_runner import TPUModelRunner

logger = init_logger(__name__)


class TPUWorker:

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

    def init_device(self):
        os.environ["PJRT_DEVICE"] = "TPU"
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        init_tpu_worker_distributed_environment(self.parallel_config,
                                                self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

        # Device initialization should happen after initializing
        # the distributed runtime.
        self.device = xm.xla_device()
        self.device_config.device = self.device

        # Set random seed.
        set_random_seed(self.model_config.seed)
        xm.set_rng_state(self.model_config.seed, self.device)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        # NOTE(woosuk): Usually, we compile 10-15 graphs for prefill and
        # 30-40 graphs for decode. 128 is an arbitrary safe number.
        torch._dynamo.config.cache_size_limit = 128
        # Use persistent cache to avoid XLA recompilation.
        # NOTE(woosuk): Set per-rank cache path since different ranks
        # can have slightly different XLA graphs.
        world_size = self.parallel_config.world_size
        rank = xr.global_ordinal()
        per_rank_path = os.path.join(envs.VLLM_XLA_CACHE_PATH,
                                     f"tp{world_size}_rank{rank}")
        xr.initialize_cache(per_rank_path, readonly=False)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = TPUModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                dtype = layer_spec.dtype

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                tpu_k_cache = torch.tensor([], dtype=dtype, device=self.device)
                tpu_v_cache = torch.tensor([], dtype=dtype, device=self.device)

                kv_caches[layer_name] = (tpu_k_cache, tpu_v_cache)
            else:
                raise NotImplementedError

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches)

        self.model_runner.dummy_run(
            runner_kv_caches,
            num_tokens=self.scheduler_config.max_num_batched_tokens,
        )

        # Synchronize before measuring the memory usage.
        xm.wait_device_ops()

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        m = xm.get_memory_info(self.device)
        total_memory_size = m["bytes_limit"]
        profiled = m["peak_bytes_used"]  # Weights + intermediate activations.

        # Calculate the TPU KV cache size based on profiling.
        usable_memory_size = int(total_memory_size *
                                 self.cache_config.gpu_memory_utilization)
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)

        return int(tpu_kv_cache_bytes)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_tpu_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""

    # NOTE(woosuk): This is just to initialize the TP group and broadcast
    # the input objects on CPU. The all-reduce and all-gather ops on TPU
    # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
    # own context.
    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="gloo",
    )
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
