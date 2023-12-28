import random

import pytest
import torch

from vllm._C import cache_ops

DTYPES = [torch.half]
NUM_TOKENS = [1]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [3]  # Arbitrary values for testing
HEAD_SIZES = [80]
BLOCK_SIZES = [16]
NUM_BLOCKS = [1]  # Arbitrary values for testing
NUM_MAPPINGS = [1]  # Arbitrary values for testing
USE_FP8_KV_CACHE = [True, False]
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
                cloned_key_cache[dst].copy_(cloned_key_cache[src])
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst].copy_(cloned_value_cache[src])

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
@pytest.mark.parametrize("use_fp8_kv_cache", USE_FP8_KV_CACHE)
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
    use_fp8_kv_cache: bool,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device="cuda")

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    cache_dtype = dtype if not use_fp8_kv_cache else torch.uint8
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, cache_dtype,
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
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    if use_fp8_kv_cache:
        # Convert key & value to fp8
        key_quant = torch.empty_like(reshaped_key, dtype=cache_dtype)
        value_quant = torch.empty_like(value, dtype=cache_dtype)
        cache_ops.convert_fp8(reshaped_key, key_quant)
        cache_ops.convert_fp8(value, value_quant)
        reshaped_key = key_quant
        value = value_quant
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    # print("key_cache:", key_cache)
    # print("cloned_key_cache", cloned_key_cache)
    # print("max diff: ", torch.max(key_cache - cloned_key_cache))
    if not use_fp8_kv_cache:
        assert torch.allclose(key_cache, cloned_key_cache)
        assert torch.allclose(value_cache, cloned_value_cache)
    else:
        assert torch.equal(key_cache, cloned_key_cache)
        assert torch.equal(value_cache, cloned_value_cache)
    # mask = torch.isclose(converted_value_cache, cloned_value_cache, rtol=1e-5, atol=1e-8, equal_nan=False)
    # mask = ~mask
    # indices = torch.nonzero(mask).cpu()
    # for index in indices:
    #     print(f"Index: {tuple(index.numpy())}, Converted Value: {converted_value_cache[index]}, Cloned Value: {cloned_value_cache[index]}")



@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_convert_fp8(
    kv_cache_factory,
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

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    converted_key_cache = torch.empty_like(key_cache, dtype=torch.uint8)
    converted_value_cache = torch.empty_like(value_cache, dtype=torch.uint8)
    # Quantize to fp8.
    cache_ops.convert_fp8(key_cache, value_cache, converted_key_cache,
                          converted_value_cache)
    # Dequantize back to dtype.
    cache_ops.convert_fp8(converted_key_cache, converted_value_cache,
                          key_cache, value_cache)

    # print("key_cache:", key_cache)
    # print("cloned_key_cache", cloned_key_cache)
    # absolute_error = torch.abs(value_cache - cloned_value_cache)
    # max_absolute_error = torch.max(absolute_error).item()

    # denominator = torch.maximum(torch.abs(value_cache),
    #                             torch.abs(cloned_value_cache))
    # relative_error = torch.where(denominator == 0,
    #                              torch.zeros_like(value_cache),
    #                              absolute_error / denominator)
    # max_relative_error = torch.max(relative_error).item()
    # print("max_absolute_error: ", max_absolute_error)
    # print("max_relative_error: ", max_relative_error)
    # NOTE(zhaoyang-star): FP8 will introduce quantization error, specifically fp8e5m2 data format.
    # Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(key_cache, cloned_key_cache, atol=1e-3, rtol=1e-1)
    assert torch.allclose(value_cache,
                          cloned_value_cache,
                          atol=1e-3,
                          rtol=1e-1)
