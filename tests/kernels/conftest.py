from typing import List, Tuple

import pytest
import torch


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.randn(size=key_cache_shape,
                                dtype=dtype,
                                device='cuda')
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.randn(size=value_cache_shape,
                                  dtype=dtype,
                                  device='cuda')
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches
