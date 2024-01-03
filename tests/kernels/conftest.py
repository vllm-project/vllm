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
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device=device)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches
