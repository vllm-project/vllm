from threadpoolctl import threadpool_info
from pprint import pprint

import random
import torch
from benchmark import KernelBenchmark
from vllm.attention_ops import single_query_cached_kv_attention


class SingleCachedAttentionBench(KernelBenchmark):

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
        num_kv_heads: int = None,
    ) -> None:
        super().__init__(loop_time)
        self.block_size = block_size
        qkv = torch.empty(num_tokens,
                          3,
                          num_heads,
                          head_size,
                          dtype=dtype,
                          device=device)
        qkv.uniform_(-1e-3, 1e-3)
        self.query, _, _ = qkv.unbind(dim=1)

        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_block_shape = (num_heads, head_size // x, block_size, x)
        self.key_cache = torch.empty(size=(num_blocks, *key_block_shape),
                                     dtype=dtype,
                                     device=device)
        self.key_cache.uniform_(-1e-3, 1e-3)
        value_block_shape = (num_heads, head_size, block_size)
        self.value_cache = torch.empty(size=(num_blocks, *value_block_shape),
                                       dtype=dtype,
                                       device=device)
        self.value_cache.uniform_(-1e-3, 1e-3)

        context_lens = [random.randint(1, 4096) for _ in range(num_tokens)]
        self.max_context_len = max(context_lens)
        self.context_lens = torch.tensor(context_lens,
                                         dtype=torch.int,
                                         device=device)

        self.max_num_blocks_per_seq = (self.max_context_len + block_size -
                                       1) // block_size
        block_tables = []
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1)
                for _ in range(self.max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        self.block_tables = torch.tensor(block_tables,
                                         dtype=torch.int,
                                         device=device)
        head_mapping = torch.arange(num_heads,
                                    dtype=torch.int32,
                                    device=device)

        self.scale = float(1.0 / (head_size**0.5))

        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert num_heads % num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device=device),
            self.num_queries_per_kv)

        self.output = torch.empty(num_tokens,
                                  num_heads,
                                  head_size,
                                  dtype=dtype,
                                  device=device)

    def _run(self):
        single_query_cached_kv_attention(
            self.output,
            self.query,
            self.key_cache,
            self.value_cache,
            self.head_mapping,
            self.scale,
            self.block_tables,
            self.context_lens,
            self.block_size,
            self.max_context_len,
            None,  # ALiBi slopes.
        )


bench = SingleCachedAttentionBench(10, 32, 32, 256, 16, 1024, torch.float32,
                                   torch.device('cpu'), 16)
bench.execute()

# SingleCachedAttentionBench(10, 32, 32, 256, 16, 1024, torch.float32, torch.device('cpu'), 16)
# Scalar: 851373304 ns
# Parallel: 70520607.25 ns 10x

pprint(threadpool_info())
