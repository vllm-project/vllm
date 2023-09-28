import random

import pytest
import torch

from vllm import cache_ops

DTYPES = [
    # torch.half, 
    # torch.bfloat16, 
    torch.float
]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
NUM_LAYERS = [5]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [
    8, 
    16, 
    32,
]
NUM_BLOCKS = [1024]  # Arbitrary values for testing
NUM_MAPPINGS = [32, 256]  # Arbitrary values for testing
SEEDS = [0]


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_copy_blocks(
    kv_cache_factory,
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping = {}
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping[src] = [dst1, dst2]

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, dtype, seed)

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Call the copy blocks kernel.
    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    # Run the reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key_cache in cloned_key_caches:
                cloned_key_cache[dst] = cloned_key_cache[src]
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst] = cloned_value_cache[src]

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches,
                                               cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create a random slot mapping.
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

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode='floor')
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_quantized(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    k_scale: float = 3.0,
    k_zp: float = 0.0,
    v_scale: float = 3.0,
    v_zp: float = 0.0,
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

    x = 16 // torch.tensor([], dtype=torch.int8).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache = torch.randint(-10, 10, size=key_cache_shape, dtype=torch.int8, device='cuda') ## change to int8
    cloned_key_cache = key_cache.clone()

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_cache = torch.randint(-10, 10, size=value_cache_shape,
                              dtype=torch.int8, ## change to int8
                              device='cuda')
    cloned_value_cache = value_cache.clone()

    cache_ops.reshape_and_cache_quantized(key, value, key_cache, value_cache,
                                          slot_mapping, k_scale, k_zp, v_scale, v_zp)
    lower_bound, upper_bound = torch.tensor([-128.0], dtype=dtype, device='cuda'), torch.tensor([127.0], dtype=dtype, device='cuda')
    ## quantize and store here
    ## quantize and store here
    quantized_key = key.reshape(num_tokens, num_heads, head_size // x, x)
    quantized_key = quantized_key.to(torch.float32)
    quantized_key = torch.maximum(lower_bound, torch.minimum(upper_bound, (quantized_key - k_zp) / k_scale))
    quantized_key = torch.round(quantized_key)
    quantized_key = quantized_key.to(torch.int8) ## change to int8

    quantized_value = value.to(torch.float32)
    quantized_value = torch.maximum(lower_bound, torch.minimum(upper_bound, (quantized_value - v_zp) / v_scale))
    quantized_value = torch.round(quantized_value)
    quantized_value = quantized_value.to(torch.int8)

    for i in range(num_tokens):
        block_idx = torch.div(slot_mapping[i],
                              block_size,
                              rounding_mode='floor')
        block_offset = slot_mapping[i] % block_size
        cloned_key_cache[block_idx, :, :, block_offset, :] = quantized_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = quantized_value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)
