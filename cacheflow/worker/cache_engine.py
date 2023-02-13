from typing import Dict, List, Tuple

import torch

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
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def allocate_gpu_cache(self) -> List[List[KVCache]]:
        gpu_cache: List[List[KVCache]] = []
        for _ in range(self.num_layers):
            layer_cache: List[KVCache] = []
            for _ in range(self.num_heads):
                key_blocks = torch.empty(
                    (self.num_gpu_blocks, self.block_size * self.head_size),
                    dtype=self.dtype,
                    device=self.gpu_id,
                )
                value_blocks = torch.empty(
                    (self.num_gpu_blocks, self.block_size * self.head_size),
                    dtype=self.dtype,
                    device=self.gpu_id,
                )
                layer_cache.append((key_blocks, value_blocks))
            gpu_cache.append(layer_cache)
        return gpu_cache

    def allocate_cpu_cache(self) -> List[List[KVCache]]:
        cpu_cache: List[List[KVCache]] = []
        for _ in range(self.num_layers):
            layer_cache: List[KVCache] = []
            for _ in range(self.num_heads):
                key_blocks = torch.empty(
                    (self.num_cpu_blocks, self.block_size * self.head_size),
                    dtype=self.dtype,
                    pin_memory=True,
                )
                value_blocks = torch.empty(
                    (self.num_cpu_blocks, self.block_size * self.head_size),
                    dtype=self.dtype,
                    pin_memory=True,
                )
                layer_cache.append((key_blocks, value_blocks))
            cpu_cache.append(layer_cache)
        return cpu_cache

    def copy(self, src_to_dst: Dict[int, int]) -> None:
        for event in self.events:
            pass

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for event in self.events:
            pass

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        for event in self.events:
            pass
