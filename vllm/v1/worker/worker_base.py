"""A GPU worker class."""
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, get_dtype_size
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.model_runner_base import ModelRunnerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


class WorkerBase:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
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

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

        # Initialized by the specific platform
        self.model_runner: Optional[ModelRunnerBase] = None

    def init_device(self):
        raise NotImplementedError()

    def load_model(self) -> None:
        assert self.model_runner is not None
        self.model_runner.load_model()

    def determine_available_memory(self) -> int:
        raise NotImplementedError()

    def compile_or_warm_up_model(self) -> None:
        assert self.model_runner is not None

        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        assert self.model_runner is not None
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> KVCacheSpec:
        assert self.model_runner is not None
        return self.model_runner.get_kv_cache_spec()

    def initialize_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        assert self.model_runner is not None
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        raise NotImplementedError()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def get_cache_block_size(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_layers_by_block_type(
        parallel_config, LayerBlockType.attention)

    key_cache_block = cache_config.block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
