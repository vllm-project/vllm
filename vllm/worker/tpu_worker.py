from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import torch

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.tpu_model_runner import TPUModelRunner
from vllm.worker.worker_base import LoraNotSupportedWorkerBase
from vllm.utils import get_dtype_size, STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)


class TPUWorker(LoraNotSupportedWorkerBase):
    """A worker class that executes (a partition of) the model on a CPU socket.

    Each worker is associated with a single CPU socket. The worker is 
    responsible for maintaining the KV cache and executing the model on the 
    CPU. In case of distributed inference, each worker is assigned a partition
    of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        vision_language_config: Optional[VisionLanguageConfig],
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.vision_language_config = vision_language_config
        assert self.device_config.device_type == "tpu"

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner = TPUModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            vision_language_config=vision_language_config)
        self.tpu_cache = None

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)
        # TODO: JAX

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_tpu_blocks = 2000
        return num_tpu_blocks, 0

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.block_size = self.cache_config.block_size

        dtype = _torch_dtype_to_jax(self.cache_dtype)
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        self.tpu_cache = []
        for _ in range(num_layers):
            key_cache = jnp.zeros(
                (num_kv_heads, num_gpu_blocks * self.block_size, head_size),
                dtype=dtype)
            value_cache = jnp.zeros_like(key_cache)
            self.tpu_cache.append((key_cache, value_cache))
        self.model_runner.block_size = self.block_size
        self._warmup_model()

    def _warmup_model(self) -> None:
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
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        assert seq_group_metadata_list is not None
        num_seq_groups = len(seq_group_metadata_list)
        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None

        # Currently, TPUWorker does not support swapping.
        # TODO(woosuk): Support block copying.
        assert len(blocks_to_swap_in) == 0
        assert len(blocks_to_swap_out) == 0
        assert len(blocks_to_copy) == 0

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output, kv_caches = self.model_runner.execute_model(
            seq_group_metadata_list, self.tpu_cache)
        self.tpu_cache = kv_caches
        return output


def _torch_dtype_to_jax(dtype: torch.dtype) -> jnp.dtype:
    mapping = {
        torch.float32: jnp.float32,
        torch.float16: jnp.float16,
        torch.bfloat16: jnp.bfloat16,
    }
    return mapping[dtype]
