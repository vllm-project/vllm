"""A TPU worker class."""

import os
from typing import TYPE_CHECKING, Tuple

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.tpu_model_runner import TPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class TPUWorker:

    def __init__(self, vllm_config: VllmConfig, local_rank: int, rank: int,
                 distributed_init_method: str):
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

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

    def initialize(self):
        os.environ["PJRT_DEVICE"] = "TPU"
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # NOTE: This is just to initialize the TP group and broadcast
        # the input objects on CPU. The all-reduce and all-gather ops on TPU
        # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
        # own context.
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size)

        # Device initialization should happen after initializing the distributed
        # runtime.
        self.device = xm.xla_device()
        self.device_config.device = self.device

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = TPUModelRunner(self.vllm_config)

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

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """

        self.model_runner.profile_run()

        # Synchronize before measuring the memory usage.
        xm.wait_device_ops()

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        m = xm.get_memory_info(self.device)
        total_tpu_memory = m["bytes_limit"]
        peak_memory = m[
            "peak_bytes_used"]  # Weights + intermediate activations.
        logger.debug("Peak Used: %sGB", peak_memory // 1024 // 1024 // 1024)
        logger.debug("Total Memory: %sGB",
                     total_tpu_memory // 1024 // 1024 // 1024)

        cache_block_size = _get_cache_block_size(self.cache_config,
                                                 self.model_config,
                                                 self.parallel_config)
        num_tpu_blocks = int(
            (total_tpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_tpu_blocks = (max(num_tpu_blocks, 0) // 8) * 8
        return num_tpu_blocks, 0

    def initialize_cache(self, num_tpu_blocks: int) -> None:
        """Allocate TPU and CPU KV cache with the specified number of blocks."""

        if num_tpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_tpu_blocks
        max_model_len = self.model_config.max_model_len
        if max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.model_runner.initialize_kv_cache(num_tpu_blocks)

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        xm.mark_step()
        xm.wait_device_ops()
        m = xm.get_memory_info(self.device)
        peak_memory = m[
            "peak_bytes_used"]  # Weights + intermediate activations.
        logger.debug("Peak GB Used Post KV Cache: %sGB",
                     peak_memory // 1024 // 1024 // 1024)

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        output = self.model_runner.execute_model(scheduler_output)
        # TODO(woosuk): Send the output to the engine process.
        return output


# TODO: this is a duplicate.
def _get_cache_block_size(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_attention_layers(
        parallel_config)

    key_cache_block = cache_config.block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
