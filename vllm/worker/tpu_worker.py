import os
from typing import List, Optional, Tuple

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.worker.tpu_model_runner import TPUModelRunner
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

logger = init_logger(__name__)


class TPUWorker(LoraNotSupportedWorkerBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        vision_language_config: Optional[VisionLanguageConfig],
        local_rank: int,
        rank: int,
        distributed_init_method: str,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.vision_language_config = vision_language_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        assert self.device_config.device_type == "tpu"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner = TPUModelRunner(model_config, parallel_config,
                                           scheduler_config, device_config,
                                           cache_config, load_config,
                                           vision_language_config)

    def init_device(self) -> None:
        os.environ["PJRT_DEVICE"] = "TPU"
        self.device = xm.xla_device()
        self.device_config.device = self.device
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # NOTE(woosuk): This is just a hack to initialize the TP group.
        # This cannot perform the actual communication ops.
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

        # Set random seed.
        set_random_seed(self.model_config.seed)
        xm.set_rng_state(self.model_config.seed, self.device)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        # NOTE(woosuk): Usually, we compile 10-15 graphs for prefill and
        # 30-40 graphs for decode. 128 is an arbitrary safe number.
        torch._dynamo.config.cache_size_limit = 128
        # Use persistent cache to avoid XLA recompilation.
        # NOTE(woosuk): This does not completely eliminate the recompilation
        # overhead because dynamo does not cache the compiled results.
        xr.initialize_cache(os.path.expanduser(envs.VLLM_XLA_CACHE_PATH),
                            readonly=False)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

        kv_caches = [(None, None) for _ in range(num_layers)]
        self.model_runner._dummy_run(
            batch_size=1,
            seq_len=self.scheduler_config.max_num_batched_tokens,
            kv_caches=kv_caches,
            is_prompt=True,
        )
        # Synchronize before measuring the memory usage.
        xm.wait_device_ops()

        m = xm.get_memory_info(self.device)
        program_size = 1024 * 1024 * 1024  # 1GB
        free_bytes = max(m["bytes_limit"] - m["bytes_used"] - program_size, 0)
        kv_cache_bytes = int(free_bytes *
                             self.cache_config.gpu_memory_utilization)
        kv_cache_dtype_btyes = get_dtype_size(self.cache_dtype)
        block_size = self.cache_config.block_size
        num_tpu_blocks = (kv_cache_bytes //
                          (kv_cache_dtype_btyes * block_size * num_layers * 2 *
                           head_size * num_kv_heads))
        num_tpu_blocks = (num_tpu_blocks // 8) * 8  # Round down to 8.
        return num_tpu_blocks, 0

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.block_size = self.cache_config.block_size

        dtype = self.cache_dtype
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        self.tpu_cache = []
        tpu_cache_shape = self.model_runner.attn_backend.get_kv_cache_shape(
            num_gpu_blocks, self.block_size, num_kv_heads, head_size)
        for _ in range(num_layers):
            key_cache = torch.zeros(tpu_cache_shape,
                                    dtype=dtype,
                                    device=self.device)
            value_cache = torch.zeros_like(key_cache)
            self.tpu_cache.append((key_cache, value_cache))
        self._warmup_model()

    def _warmup_model(self) -> None:
        # FIXME(woosuk): Here we are abusing `enforce_eager` which is defined
        # for CUDA graphs. We should refactor this part.
        if not self.model_config.enforce_eager:
            # Warm up the model with all possible input shapes so that
            # compilation never happens during the actual execution.
            # This may take ~30 mins for the first run and ~20 mins for the
            # subsequent runs.
            # If `enforce_eager` is True, the ahead-of-time compilation is
            # skipped and the compilation happens during the actual execution,
            # which is bad for performance but useful for development.
            self.model_runner.warmup_model(self.tpu_cache)

    def get_cache_block_size_bytes(self) -> int:
        head_size = self.model_config.get_head_size()
        num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        num_layers = self.model_config.get_num_layers(self.parallel_config)

        key_cache_block = self.cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = get_dtype_size(self.cache_dtype)
        return dtype_size * total

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if execute_model_req is None:
            return []

        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        num_seq_groups = len(seq_group_metadata_list)
        if num_seq_groups == 0:
            return []

        # Currently, TPUWorker does not support swapping.
        # TODO(woosuk): Support block copying.
        assert len(execute_model_req.blocks_to_swap_in) == 0, (
            "Swapping is not supported for the TPU backend.")
        assert len(execute_model_req.blocks_to_swap_out) == 0, (
            "Swapping is not supported for the TPU backend.")
        assert len(execute_model_req.blocks_to_copy) == 0

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.tpu_cache)
        return [output]
