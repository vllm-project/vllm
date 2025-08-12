import random
from typing import Tuple

import pytest
import torch

from vllm._C import cache_ops
from vllm.utils import is_hip

COPYING_DIRECTION = [('cuda', 'cpu'), ('cuda', 'cuda'), ('cpu', 'cuda')]
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [8, 16, 32]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
KV_CACHE_DTYPE = ["auto", "fp8"]


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
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
    kv_cache_dtype: str,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
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
                                                head_size, kv_cache_dtype,
                                                dtype, seed, device)

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
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
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
    device: str,
    kv_cache_dtype: str,
) -> None:
    if not is_hip() and kv_cache_dtype == "fp8":
        pytest.skip()  # This test is not tuned for e5m2 cuda precision
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        cache_ops.convert_fp8(key_cache, cloned_key_cache)
        cloned_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        cache_ops.convert_fp8(value_cache, cloned_value_cache)
    else:
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

    # Using default kv_scale
    kv_scale = 1.0

    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping, kv_cache_dtype, kv_scale)

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        cache_ops.convert_fp8(key_cache, result_key_cache)
        result_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        cache_ops.convert_fp8(value_cache, result_value_cache)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    if kv_cache_dtype == "fp8":
        assert torch.allclose(result_key_cache,
                              cloned_key_cache,
                              atol=0.001,
                              rtol=0.1)
        assert torch.allclose(result_value_cache,
                              cloned_value_cache,
                              atol=0.001,
                              rtol=0.1)
    else:
        assert torch.allclose(key_cache, cloned_key_cache)
        assert torch.allclose(value_cache, cloned_value_cache)


@pytest.mark.parametrize("direction", COPYING_DIRECTION)
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_swap_blocks(
    kv_cache_factory,
    direction: Tuple[str, str],
    num_mappings: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and "cpu" in direction:
        pytest.skip()
    if not is_hip() and kv_cache_dtype == "fp8":
        pytest.skip()  # This test is not tuned for e5m2 cuda precision
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    src_device = device if direction[0] == "cuda" else 'cpu'
    dst_device = device if direction[1] == "cuda" else 'cpu'

    src_blocks = random.sample(range(num_blocks), num_mappings)
    # For the same device, mapping must not overlap
    if src_device == dst_device:
        remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
        dst_blocks = random.sample(remaining_blocks, num_mappings)
    else:
        dst_blocks = random.sample(range(num_blocks), num_mappings)

    block_mapping = dict(zip(src_blocks, dst_blocks))

    # Create the KV caches on the first device.
    src_key_caches, src_value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype,
        seed, src_device)

    # Create the KV caches on the second device.
    dist_key_caches, dist_value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype,
        seed, dst_device)

    src_key_caches_clone = src_key_caches[0].clone()
    src_value_caches_clone = src_value_caches[0].clone()

    # Call the swap_blocks kernel.
    cache_ops.swap_blocks(src_key_caches[0], dist_key_caches[0], block_mapping)
    cache_ops.swap_blocks(src_value_caches[0], dist_value_caches[0],
                          block_mapping)

    for src, dst in block_mapping.items():
        assert torch.allclose(src_key_caches_clone[src].cpu(),
                              dist_key_caches[0][dst].cpu())
        assert torch.allclose(src_value_caches_clone[src].cpu(),
                              dist_value_caches[0][dst].cpu())


@pytest.mark.skipif(not is_hip(), reason="FP8 conversion test requires e4m3")
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fp8_conversion(
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    low = -224.0
    high = 224.0
    shape = (num_blocks, num_heads, head_size, block_size)
    cache = torch.empty(shape, dtype=dtype, device=device)
    cache.uniform_(low, high)

    cache_fp8 = torch.empty_like(cache, dtype=torch.uint8)
    cache_ops.convert_fp8(cache, cache_fp8)

    converted_cache = torch.empty_like(cache)
    cache_ops.convert_fp8(cache_fp8, converted_cache)

    assert torch.allclose(cache, converted_cache, atol=0.001, rtol=0.1)
