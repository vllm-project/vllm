import random

import torch

from vllm import cache_ops


@torch.inference_mode()
def run_copy_blocks(
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    # Generate random block mappings.
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, num_mappings)
    block_mapping = {src: [dst] for src, dst in zip(src_blocks, dst_blocks)}

    # Create the KV cache.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.randn(size=key_cache_shape,
                                dtype=dtype,
                                device='cuda')
        key_caches.append(key_cache)
    cloned_key_caches = []
    for key_cache in key_caches:
        cloned_key_caches.append(key_cache.clone())

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.randn(size=value_cache_shape,
                                  dtype=dtype,
                                  device='cuda')
        value_caches.append(value_cache)
    cloned_value_caches = []
    for value_cache in value_caches:
        cloned_value_caches.append(value_cache.clone())

    # Call the copy blocks kernel.
    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    # Reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for key_cache, cloned_key_cache in zip(key_caches,
                                                   cloned_key_caches):
                cloned_key_cache[dst] = cloned_key_cache[src]
            for value_cache, cloned_value_cache in zip(value_caches,
                                                       cloned_value_caches):
                cloned_value_cache[dst] = cloned_value_cache[src]

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches,
                                               cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)


@torch.inference_mode()
def run_reshape_and_cache(
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

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    _, key, value = qkv.unbind(dim=1)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(size=key_cache_shape, dtype=dtype, device='cuda')
    cloned_key_cache = key_cache.clone()

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_cache = torch.randn(size=value_cache_shape,
                              dtype=dtype,
                              device='cuda')
    cloned_value_cache = value_cache.clone()

    cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping)

    for i in range(num_tokens):
        reshaped_key = key.reshape(num_tokens, num_heads, head_size // x, x)
        block_idx = torch.div(slot_mapping[i],
                              block_size,
                              rounding_mode='floor')
        block_offset = slot_mapping[i] % block_size
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)


@torch.inference_mode()
def run_gather_cached_kv(
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

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    _, key, value = qkv.unbind(dim=1)

    qkv_clone = qkv.clone()
    _, cloned_key, cloned_value = qkv_clone.unbind(dim=1)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(size=key_cache_shape, dtype=dtype, device='cuda')

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_cache = torch.randn(size=value_cache_shape,
                              dtype=dtype,
                              device='cuda')

    cache_ops.gather_cached_kv(key, value, key_cache, value_cache,
                               slot_mapping)

    # Reference implementation.
    for i in range(num_tokens):
        reshaped_key = cloned_key.reshape(num_tokens, num_heads,
                                          head_size // x, x)
        block_idx = torch.div(slot_mapping[i],
                              block_size,
                              rounding_mode='floor')
        block_offset = slot_mapping[i] % block_size
        reshaped_key[i] = key_cache[block_idx, :, :, block_offset, :]
        cloned_value[i] = value_cache[block_idx, :, :, block_offset]

    assert torch.allclose(key, cloned_key)
    assert torch.allclose(value, cloned_value)


def test_copy_blocks() -> None:
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        run_copy_blocks(num_mappings=23,
                        num_layers=7,
                        num_heads=17,
                        head_size=16,
                        block_size=8,
                        num_blocks=1024,
                        dtype=dtype)


def test_reshape_and_cache() -> None:
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        run_reshape_and_cache(num_tokens=3,
                              num_heads=2,
                              head_size=16,
                              block_size=8,
                              num_blocks=2,
                              dtype=dtype)


def test_gather_cached_kv() -> None:
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        run_gather_cached_kv(num_tokens=3,
                             num_heads=2,
                             head_size=16,
                             block_size=8,
                             num_blocks=2,
                             dtype=dtype)
