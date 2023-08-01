from threadpoolctl import threadpool_info
from pprint import pprint

import torch
from benchmark import KernelBenchmark
from vllm import pos_encoding_ops


class PosEncodingBench(KernelBenchmark):

    def __init__(self, loop_time, num_tokens: int, num_heads: int,
                 head_size: int, max_position: int, rotary_dim: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(loop_time)
        base: int = 10000
        self.positions = torch.randint(0,
                                       max_position, (num_tokens, ),
                                       device=device)
        query = torch.randn(num_tokens,
                            num_heads * head_size,
                            dtype=dtype,
                            device=device)
        key = torch.randn(num_tokens,
                          num_heads * head_size,
                          dtype=dtype,
                          device=device)
        # Create the rotary embedding.
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        self.head_size = head_size
        self.cos_sin_cache = torch.cat((cos, sin), dim=-1)
        self.cos_sin_cache = self.cos_sin_cache.to(dtype=dtype, device=device)
        self.out_query = query.clone()
        self.out_key = key.clone()

    def _run(self):
        for i in range(self.loop_time):
            pos_encoding_ops.rotary_embedding_neox(self.positions,
                                                   self.out_query,
                                                   self.out_key,
                                                   self.head_size,
                                                   self.cos_sin_cache)


bench = PosEncodingBench(10,
                         num_tokens=4096,
                         num_heads=5,
                         head_size=128,
                         max_position=8192,
                         rotary_dim=128,
                         dtype=torch.float32,
                         device=torch.device("cpu"))
bench.execute()

pprint(threadpool_info())
