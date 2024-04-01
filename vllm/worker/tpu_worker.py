"""A TPU worker class."""
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch_xla.core.xla_model as xm

from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.model_executor import set_random_seed
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


class TPUWorker:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        assert self.device_config.device_type == "tpu"
        self.device_config.device = xm.xla_device()
        self.device = self.device_config.device

        self.model_runner = TPUModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            lora_config=self.lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config)
        self.cache_config = None
        self.tpu_cache = None

    def init_device(self) -> None:
        # Set random seed.
        self._set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        self._set_random_seed(self.model_config.seed)

    @torch.inference_mode()
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

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.tpu_cache)
        return output

    def allocate_kv_cache(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            cache_config.num_gpu_blocks, cache_config.block_size, self.num_heads, self.head_size)
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            device=self.device))
        self.tpu_cache = kv_cache

    def _set_random_seed(self, seed: int) -> None:
        xm.set_rng_state(seed, device=self.device)
        set_random_seed(seed)


class CacheEngine:

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_tpu_blocks = cache_config.num_gpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        self.device = device_config.device

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.dtype)

        # Initialize the cache.
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            self.num_tpu_blocks, self.block_size, self.num_heads, self.head_size)
        self.tpu_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            self.tpu_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            device=self.device))

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        raise NotImplementedError(
            "Copying blocks is not supported on TPU backend.")

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
