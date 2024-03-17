"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Skip initializing CUDA stream and buffer for Neuron backend.
        if is_neuron():
            return

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        kv_cache_size = (2 * self.num_gpu_blocks * self.block_size *
                         self.num_heads * self.head_size)
        gpu_cache = [
            torch.empty(kv_cache_size, dtype=self.dtype, device="cuda")
            for _ in range(self.num_layers)
        ]
        return gpu_cache

    def allocate_cpu_cache(self) -> List[torch.Tensor]:
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")

        kv_cache_size = (2 * self.num_cpu_blocks * self.block_size *
                         self.num_heads * self.head_size)
        cpu_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            cpu_cache.append(
                torch.empty(kv_cache_size,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device="cpu"))
        return cpu_cache

    # FIXME(woosuk)
    def _swap(
        self,
        src: List[torch.Tensor],
        dst: List[torch.Tensor],
        src_to_dst: Dict[int, int],
    ) -> None:
        from vllm._C import cache_ops

        for i in range(self.num_layers):
            src_key_cache, src_value_cache = src[i]
            dst_key_cache, dst_value_cache = dst[i]
            # Copy the key blocks.
            cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
            # Copy the value blocks.
            cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    # FIXME(woosuk)
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        from vllm._C import cache_ops

        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

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
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
