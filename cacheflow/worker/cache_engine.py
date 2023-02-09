from typing import List, Tuple

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
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        self.dtype = dtype

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the streams.
        self.copy_stream = torch.cuda.Stream(device=gpu_id)
        self.swap_stream = torch.cuda.Stream(device=gpu_id)
        assert self.copy_stream != self.swap_stream
        current_stream = torch.cuda.current_stream(device=gpu_id)
        assert self.copy_stream != current_stream
        assert self.swap_stream != current_stream

        # Initialize the events for synchronization.

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

    def copy(
        self,
        src_block_numbers: List[int],
        dst_block_numbers: List[int],
    ) -> None:
        for layer in range(self.num_layers):
            # TODO: Call the COPY op.
            pass

    def swap_out(
        self,
        gpu_block_numbers: List[int],
        cpu_block_numbers: List[int],
    ) -> None:
        for layer in range(self.num_layers):
            # TODO: Call the SWAP_OUT op on the swap stream.
            pass

    def swap_in(
        self,
        gpu_block_numbers: List[int],
        cpu_block_numbers: List[int],
    ) -> None:
        for layer in range(self.num_layers):
            # TODO: Call the SWAP_IN op on the swap stream.
            pass
