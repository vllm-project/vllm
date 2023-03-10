from typing import Dict, List, Tuple

import torch
from cacheflow import cache_ops

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CacheEngine:

    def __init__(
        self,
        worker_id: int,
        gpu_id: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
    ) -> None:
        if head_size % 16 != 0:
            raise ValueError(
                f'head_size ({head_size}) must be a multiple of 16.')

        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.dtype = dtype

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream(device=gpu_id)
        assert self.cache_stream != torch.cuda.current_stream(device=gpu_id)
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(num_layers)]

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
                device=self.gpu_id,
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device=self.gpu_id,
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=True,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=True,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(
                    src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(
                    src_value_cache, dst_value_cache, src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def _copy(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dsts: Dict[int, List[int]],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.copy_blocks(
                    src_key_cache, dst_key_cache, src_to_dsts)
                # Copy the value blocks.
                cache_ops.copy_blocks(
                    src_value_cache, dst_value_cache, src_to_dsts)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self._copy(self.gpu_cache, self.gpu_cache, src_to_dsts)
