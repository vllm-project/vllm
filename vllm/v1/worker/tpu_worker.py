# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TPU worker class."""

import os
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.platforms.tpu import USE_TPU_INFERENCE
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.utils import bind_kv_cache

logger = init_logger(__name__)

_R = TypeVar("_R")

if not USE_TPU_INFERENCE:
    logger.info("tpu_inference not found, using vLLM's TPUWorker.")
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    import torch_xla.runtime as xr

    from vllm.v1.attention.backends.pallas import TPU_HEAD_SIZE_ALIGNMENT
    from vllm.v1.worker.tpu_model_runner import TPUModelRunner


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
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        self.original_parallel_config = None
        if self.use_spmd:
            # Under SPMD mode, distributed env is initialized as if there is
            # only one worker/device.
            self.original_parallel_config = self.parallel_config
            self.parallel_config.tensor_parallel_size = 1
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.world_size = 1
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils.import_utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Delay profiler initialization to the start of the profiling.
        # This is because in vLLM V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profiler = None
        self.profile_dir = None
        if envs.VLLM_TORCH_PROFILER_DIR and self.rank < 1:
            # For TPU, we can only have 1 active profiler session for 1 profiler
            # server. So we only profile on rank0.
            self.profile_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info(
                "Profiling enabled. Traces will be saved to: %s", self.profile_dir
            )

        if self.model_config.seed is None:
            self.model_config.seed = 0

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        os.environ["PJRT_DEVICE"] = "TPU"
        # Note: Currently the XLA compiler wrongly uses 2D ring strategy on 1D
        # ring, the xla tpu compiler flag
        # `xla_tpu_force_1d_allreduce_at_chunk_count` is a temporary solution to
        # fix this. It will be removed after the bug in XLA compiler is fixed.
        os.environ["LIBTPU_INIT_ARGS"] = (
            os.environ.get("LIBTPU_INIT_ARGS", "")
            + " --xla_tpu_force_1d_allreduce_at_chunk_count=1"
            " --xla_jf_conv_input_fusion=False"
        )
        # --xla_jf_conv_input_fusion=False is used to improve the perf of
        # quantized matmul.
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        self._init_tpu_worker_distributed_environment(
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
        )

        # Device initialization should happen after initializing
        # the distributed runtime.
        self.device = xm.xla_device()
        self.device_config.device = self.device

        # Set random seed.
        set_random_seed(self.model_config.seed)
        xm.set_rng_state(self.model_config.seed, self.device)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        # TODO (NickLucche) On gsm we compile 80+ graphs.
        # Re-evaluate limit, with MM we may get close to this limit.
        torch._dynamo.config.cache_size_limit = 128
        # Use persistent cache to avoid XLA recompilation.
        # NOTE(woosuk): Set per-rank cache path since different ranks
        # can have slightly different XLA graphs.
        world_size = self.parallel_config.world_size
        rank = xr.global_ordinal()
        # The PyTorch/XLA compilation cache uses the Torch IR to generate keys.
        # Consequently, changes in optimization flags, which affect compilation
        # results, don't change the cache key. This can result in the wrong
        # compilation being used. To prevent this, disabling the XLA compilation
        # cache during development is recommended.We can disable it by
        # `export VLLM_XLA_CACHE_PATH=`
        if envs.VLLM_XLA_CACHE_PATH:
            per_rank_path = os.path.join(
                envs.VLLM_XLA_CACHE_PATH, f"tp{world_size}_rank{rank}"
            )
            xr.initialize_cache(per_rank_path, readonly=False)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = TPUModelRunner(
            self.vllm_config, self.device, self.original_parallel_config
        )

        if rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def determine_available_memory(self) -> int:
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, AttentionSpec):
                dtype = layer_spec.dtype

                # Use an empty tensor instead of `None` to force Dynamo to pass
                # it by reference, rather by specializing on the value `None`.
                tpu_kv_cache = torch.tensor([], dtype=dtype).to(self.device)
                kv_caches[layer_name] = tpu_kv_cache
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec '{type(layer_spec)}'"
                )

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches,
        )

        # `max_num_tokens >= max_num_batched_tokens` due to padding.
        with self.model_runner.maybe_setup_dummy_loras(self.lora_config):
            self.model_runner.profile_run(self.model_runner.max_num_tokens)

        # Synchronize before measuring the memory usage.
        xm.wait_device_ops()

        # During the profiling run, the model runs without KV cache. After
        # the profiling run, the model always runs with KV cache. Here we clear
        # the dynamo cache and cached bytecode to ensure the model always has
        # one compiled bytecode. Having one FX graph/cached bytecode per
        # compiled model is required for `support_torch_compile` decorator to
        # skip dynamo guard.
        self.model_runner.reset_dynamo_cache()

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        if self.use_spmd:
            # This is a workaround for the TPU SPMD mode. The get_memory_info
            # API doesn't work with SPMD mode in PyTorch/XLA.
            # TODO: use xm.get_memory_info for SPMD once it's supported in
            # PyTorch/XLA.
            import tpu_info

            chip_type, _ = tpu_info.device.get_local_chips()
            device_usage = tpu_info.metrics.get_chip_usage(chip_type)
            total_memory_size = device_usage[0].total_memory
            current_mem = device_usage[0].memory_usage
        else:
            m = xm.get_memory_info(self.device)
            total_memory_size = m["bytes_limit"]
            current_mem = m["bytes_used"]
        # Ideally we would use profiled = m["peak_bytes_used"] to
        # get weights + activations. But there is memory used during
        # compilation / weight loading that impacts the peak and
        # there is no way to reset peak memory in XLA, So we
        # use the heuristic of 2% of weights.
        profiled = current_mem * 1.02

        # Calculate the TPU KV cache size based on profiling.
        usable_memory_size = int(
            total_memory_size * self.cache_config.gpu_memory_utilization
        )
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)
        head_size = self.model_config.get_head_size()
        if head_size > 0:
            padded_head_size = (
                cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
            )
            if padded_head_size != head_size:
                logger.warning_once("head size is padded to %d", padded_head_size)
            # We adjust the usable memory size for the KV cache to prevent OOM
            # errors, even after padding the head_size.
            tpu_kv_cache_bytes = tpu_kv_cache_bytes * head_size // padded_head_size
        return int(tpu_kv_cache_bytes)

    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output)

    def profile(self, is_start: bool = True):
        if self.rank < 1:
            if self.profile_dir is None:
                raise RuntimeError("Profiler is not enabled.")
            if is_start:
                if self.profiler is None:
                    self.profiler = xp.start_server(9012)
                xp.start_trace(self.profile_dir)
            else:
                xp.stop_trace()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def _init_tpu_worker_distributed_environment(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str | None = None,
        local_rank: int = -1,
    ) -> None:
        """Initialize the distributed environment."""
        if self.use_spmd:
            xr.use_spmd()
        # NOTE(woosuk): This is just to initialize the TP group and broadcast
        # the input objects on CPU. The all-reduce and all-gather ops on TPU
        # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
        # own context.
        parallel_config = vllm_config.parallel_config
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=distributed_init_method or "env://",
            backend=current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
        )

        ensure_kv_transfer_initialized(vllm_config)

    def shutdown(self) -> None:
        self.model_runner.ensure_kv_transfer_shutdown()

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())


if USE_TPU_INFERENCE:
    from tpu_inference.worker import TPUWorker as TpuInferenceWorker

    TPUWorker = TpuInferenceWorker  # type: ignore
