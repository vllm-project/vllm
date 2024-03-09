"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple, Union

import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)

from vllm.block import KVCache


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    @staticmethod
    def from_config(
        cache_config: CacheConfig, model_config: ModelConfig,
        parallel_config: ParallelConfig
    ) -> Union['PagedCacheEngine', 'FlashInferCacheEngine']:
        if __import__("os").getenv("VLLM_TEMP_USE_FLASH", "0") == "1":
            return FlashInferCacheEngine(cache_config, model_config,
                                    parallel_config)
        else:
            return PagedCacheEngine(cache_config, model_config,
                                    parallel_config)

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

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError
    
    def _allocate_arena(self) -> KVCache:
        raise NotImplementedError
    
    def _allocate_arena_cpu(self, pin_memory: bool) -> KVCache:
        raise NotImplementedError
    
    def allocate_gpu_cache(self) -> List[KVCache]:
        return [self._allocate_arena() for _ in range(self.num_layers)]

    def allocate_cpu_cache(self) -> List[KVCache]:
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        return [self._allocate_arena_cpu(pin_memory) for _ in range(self.num_layers)]

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)
        
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        raise NotImplementedError

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
    

class PagedCacheEngine(CacheEngine):

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig) -> None:
        super().__init__(cache_config, model_config, parallel_config)

    def get_key_block_shape(self) -> Tuple[int, ...]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, ...]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )
    
    def _allocate_arena(self) -> KVCache:
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        key_blocks = torch.empty(
            size=(self.num_gpu_blocks, *key_block_shape),
            dtype=self.dtype,
            device="cuda",
        )
        value_blocks = torch.empty(
            size=(self.num_gpu_blocks, *value_block_shape),
            dtype=self.dtype,
            device="cuda",
        )
        return (key_blocks, value_blocks)
    
    def _allocate_arena_cpu(self, pin_memory: bool) -> KVCache:
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        key_blocks = torch.empty(
            size=(self.num_cpu_blocks, *key_block_shape),
            dtype=self.dtype,
            pin_memory=pin_memory,
            device="cpu",
        )
        value_blocks = torch.empty(
            size=(self.num_cpu_blocks, *value_block_shape),
            dtype=self.dtype,
            pin_memory=pin_memory,
            device="cpu",
        )
        return (key_blocks, value_blocks)
    
    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        from vllm._C import cache_ops

        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)
    
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        from vllm._C import cache_ops

        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)


class FlashInferCacheEngine(CacheEngine):

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig) -> None:
        super().__init__(cache_config, model_config, parallel_config)

    def get_key_block_shape(self) -> Tuple[int, ...]:
        return (self.block_size, self.num_heads, self.head_size)

    def get_value_block_shape(self) -> Tuple[int, ...]:
        return self.get_key_block_shape()
    
    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        from vllm._C import cache_ops

        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                cache_ops.swap_blocks(src[i], dst[i], src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)
        
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        raise NotImplementedError

    def _allocate_arena(self) -> KVCache:
        key_block_shape = self.get_key_block_shape()
        return torch.empty(
            size=(self.num_gpu_blocks, 2, *key_block_shape),
            dtype=self.dtype,
            device="cuda")

    def _allocate_arena_cpu(self, pin_memory: bool) -> KVCache:
        key_block_shape = self.get_key_block_shape()
        return torch.empty(
            size=(self.num_cpu_blocks, 2, *key_block_shape),
            dtype=self.dtype,
            pin_memory=pin_memory,
            device="cpu")


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
