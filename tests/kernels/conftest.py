from typing import List, Tuple

import pytest
import torch

from vllm._C import cache_ops


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype,
                                device=device)
        if dtype != torch.uint8:
            key_cache.uniform_(-scale, scale)
        else:
            # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
            # it may occur Inf or NaN if we directly use torch.randint
            # to generate random data for fp8 cache.
            # For example, s.11111.00 in fp8e5m2 format repesents Inf.
            #     | E4M3        | E5M2
            #-----|-------------|-------------------
            # Inf | N/A         | s.11111.00
            # NaN | s.1111.111  | s.11111.{01,10,11}
            key_cache_tmp = torch.empty_like(key_cache, dtype=torch.float16)
            key_cache_tmp.uniform_(-scale, scale)
            cache_ops.convert_fp8(key_cache_tmp, key_cache)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device=device)
        if dtype != torch.uint8:
            value_cache.uniform_(-scale, scale)
        else:
            value_cache_tmp = torch.empty_like(value_cache,
                                               dtype=torch.float16)
            value_cache_tmp.uniform_(-scale, scale)
            cache_ops.convert_fp8(value_cache_tmp, value_cache)
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches
