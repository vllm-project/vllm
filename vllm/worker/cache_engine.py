"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Optional, Tuple

import torch

from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
from vllm.model_executor.weight_utils import kv_cache_scales_iterator
from vllm.utils import in_wsl, STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]


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
        
        # We enable cache scaling factors if and only if cache is FP8-typed
        self.use_scaling_factor = torch.tensor([], dtype=self.dtype).element_size() == 1

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Load scaling factors into the GPU cache if values are specified
        # We do not need to load them into the CPU cache because they are
        # never swapped out.
        if self.cache_config.kv_cache_scales is not None:
            self.load_kv_cache_scales(self.cache_config.kv_cache_scales)

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
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
            if self.use_scaling_factor:
                scaling_factor = torch.ones(
                    1,
                    dtype=torch.float32,
                    device='cuda',
                )
            else:
                scaling_factor = None
            gpu_cache.append((key_blocks, value_blocks, scaling_factor))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
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
            # Scale factors are not involved in the swap process and never need to reside on CPU
            scaling_factor = None
            cpu_cache.append((key_blocks, value_blocks, scaling_factor))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        from vllm._C import cache_ops

        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache, src_scaling = src[i]
                dst_key_cache, dst_value_cache, dst_scaling = dst[i]
                # We should not need to copy scaling factors, as they are equal
                # given a fixed layer
                # TODO(mattwong) Remove this once confirmed
                if self.use_scaling_factor:
                    assert torch.equal(src_scaling, dst_scaling)
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache, _ in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
    
    # Helper function to load in static KV cache scaling factors (one per layer)
    # stored in a given file. These scaling factors are assumed to not take up
    # too much space and are hence permanently resident on GPU.
    def load_kv_cache_scales(self, filename: str):
        for layer_idx, scaling_factor in kv_cache_scales_iterator(filename):
            self.gpu_cache[layer_idx][2].copy_(scaling_factor)


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
        
        use_scaling_factor = dtype_size == 1
        if use_scaling_factor:
            return dtype_size * total + num_layers * 4
        
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
