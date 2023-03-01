import random

import torch

from cacheflow import cache_ops


def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device='cuda')

    kv_shape = (num_tokens, num_heads, head_size)
    key = torch.randn(size=kv_shape, dtype=dtype, device='cuda')
    value = torch.randn(size=kv_shape, dtype=dtype, device='cuda')
    
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(size=key_cache_shape, dtype=dtype, device='cuda')
    cloned_key_cache = key_cache.clone()

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_cache = torch.randn(
        size=value_cache_shape, dtype=dtype, device='cuda')
    cloned_value_cache = value_cache.clone()

    cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    for i in range(num_tokens):
        reshaped_key = key.reshape(num_tokens, num_heads, head_size // x, x)
        block_idx = slot_mapping[i] // block_size
        block_offset = slot_mapping[i] % block_size
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)


@torch.inference_mode()
def test_cache() -> None:
    test_reshape_and_cache(
        num_tokens=3, num_heads=2, head_size=16, block_size=8, num_blocks=2,
        dtype=torch.half)


if __name__ == '__main__':
    test_cache()
