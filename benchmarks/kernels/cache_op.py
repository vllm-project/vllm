from threadpoolctl import threadpool_info
from pprint import pprint

import torch
import random
from benchmark import KernelBenchmark
from vllm.cache_ops import copy_blocks, reshape_and_cache


class CacheCopyBench(KernelBenchmark):

    def __init__(
        self,
        loop_time,
        num_mappings: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(loop_time)
        # Generate random block mappings.
        src_blocks = random.sample(range(num_blocks), num_mappings)
        remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
        dst_blocks = random.sample(remainig_blocks, num_mappings)
        self.block_mapping = {
            src: [dst]
            for src, dst in zip(src_blocks, dst_blocks)
        }

        # Create the KV cache.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (num_blocks, num_heads, head_size // x, block_size,
                           x)
        self.key_caches = []
        for _ in range(num_layers):
            key_cache = torch.randn(size=key_cache_shape,
                                    dtype=dtype,
                                    device=device)
            self.key_caches.append(key_cache)

        value_cache_shape = (num_blocks, num_heads, head_size, block_size)
        self.value_caches = []
        for _ in range(num_layers):
            value_cache = torch.randn(size=value_cache_shape,
                                      dtype=dtype,
                                      device=device)
            self.value_caches.append(value_cache)

    def _run(self):
        for i in range(self.loop_time):
            copy_blocks(self.key_caches, self.value_caches, self.block_mapping)


class CacheReshapeBench(KernelBenchmark):

    def __init__(
        self,
        loop_time,
        num_tokens: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(loop_time)
        num_slots = block_size * num_blocks
        slot_mapping = random.sample(range(num_slots), num_tokens)
        self.slot_mapping = torch.tensor(slot_mapping,
                                         dtype=torch.int,
                                         device=device)

        qkv = torch.randn(num_tokens,
                          3,
                          num_heads,
                          head_size,
                          dtype=dtype,
                          device=device)
        _, self.key, self.value = qkv.unbind(dim=1)

        self.x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (num_blocks, num_heads, head_size // self.x,
                           block_size, self.x)
        self.key_cache = torch.randn(size=key_cache_shape,
                                     dtype=dtype,
                                     device=device)

        value_cache_shape = (num_blocks, num_heads, head_size, block_size)
        self.value_cache = torch.randn(size=value_cache_shape,
                                       dtype=dtype,
                                       device=device)

    def _run(self):
        reshape_and_cache(self.key, self.value, self.key_cache,
                          self.value_cache, self.slot_mapping)


# bench = CacheCopyBench(10, 256, 8, 16, 256, 16, 1024, torch.float32, torch.device("cpu"))
# bench.execute()

# CacheCopyBench(10, 256, 8, 16, 256, 16, 1024, torch.float32, torch.device("cpu"))
# Scalar: 2731509071.375 ns
# Layer parallel: 510428213.5 ns 5.35x
# nested parallel: 434456796.5 ns 6.05x
# section parallel: 442927758 ns

# bench = CacheReshapeBench(10, 128, 64, 256, 16, 1024, torch.float32, torch.device("cpu"))
# bench.execute()

# CacheReshapeBench(10, 128, 64, 256, 16, 1024, torch.float32, torch.device("cpu"))
# Scalar: 77548817.875 ns
# Parallel: 7257660.75 ns 10x

pprint(threadpool_info())
