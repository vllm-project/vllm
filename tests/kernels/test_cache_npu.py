import random
from typing import List, Tuple

import pytest
import torch
from typing import (List, Optional, Tuple, Union)

# from vllm import _custom_ops as ops
from vllm.attention.backends.ascend import AscendAttentionBackend
from vllm.utils import get_kv_cache_torch_dtype

COPYING_DIRECTION = [('npu', 'cpu'), ('npu', 'npu'), ('cpu', 'npu')]
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]
BLOCK_SIZES = [8, 16, 32]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
NPU_DEVICES = [
    "npu:0"
]

# We assume fp8 is always enabled for testing.
# KV_CACHE_DTYPE = ["auto", "fp8"]
KV_CACHE_DTYPE = ["float"]

copy_blocks = AscendAttentionBackend.copy_blocks
swap_blocks = AscendAttentionBackend.swap_blocks


def create_kv_caches_with_random_for_npu(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch.random.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    key_cache_shape = (num_blocks, block_size, num_heads * head_size)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, block_size, num_heads * head_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", NPU_DEVICES)
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
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
    torch.set_default_device(device)
    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping: List[Tuple[int, int]] = []
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping.append((src, dst1))
        block_mapping.append((src, dst2))

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random_for_npu(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, kv_cache_dtype,
                                                dtype, seed, device)

    kv_caches = []
    for i in range(len(key_caches)):
        kv_caches.append(torch.tensor(torch.cat((key_caches[i].unsqueeze(0), value_caches[i].unsqueeze(0)), dim=0)))

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Call the copy blocks kernel.
    block_mapping_tensor = torch.tensor(block_mapping,
                                        dtype=torch.int64,
                                        device=device).view(-1, 2)
    # ops.copy_blocks(key_caches, value_caches, block_mapping_tensor)
    copy_blocks(kv_caches, block_mapping_tensor)

    # Run the reference implementation.
    for src, dst in block_mapping:
        for cloned_key_cache in cloned_key_caches:
            cloned_key_cache[dst].copy_(cloned_key_cache[src])
        for cloned_value_cache in cloned_value_caches:
            cloned_value_cache[dst].copy_(cloned_value_cache[src])

    # Compare the results.
    for kv_cache, cloned_key_cache, cloned_value_cache in zip(kv_caches, cloned_key_caches, cloned_value_caches):
        k = kv_cache[0]
        v = kv_cache[1]
        torch.testing.assert_close(k, cloned_key_cache)
        torch.testing.assert_close(v, cloned_value_cache)


@pytest.mark.parametrize("direction", COPYING_DIRECTION)
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", NPU_DEVICES)
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
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)

    src_device = device if direction[0] == "npu" else 'cpu'
    dst_device = device if direction[1] == "npu" else 'cpu'

    src_blocks = random.sample(range(num_blocks), num_mappings)
    # For the same device, mapping must not overlap
    if src_device == dst_device:
        remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
        dst_blocks = random.sample(remaining_blocks, num_mappings)
    else:
        dst_blocks = random.sample(range(num_blocks), num_mappings)

    block_mapping = list(zip(src_blocks, dst_blocks))
    block_mapping_tensor = torch.tensor(block_mapping,
                                        dtype=torch.int64,
                                        device="cpu").view(-1, 2)

    # Create the KV caches on the first device.
    src_key_caches, src_value_caches = create_kv_caches_with_random_for_npu(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype,
        seed, src_device)

    # Create the KV caches on the second device.
    dist_key_caches, dist_value_caches = create_kv_caches_with_random_for_npu(
        num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype,
        seed, dst_device)

    src_key_caches_clone = src_key_caches[0].clone()
    src_value_caches_clone = src_value_caches[0].clone()

    # Call the swap_blocks kernel.
    src_kv_caches = []
    dist_kv_caches = []
    for i in range(len(src_key_caches)):
        src_kv_caches.append(torch.tensor(torch.cat((src_key_caches[i].unsqueeze(0), src_value_caches[i].unsqueeze(0)), dim=0)))
    for i in range(len(dist_key_caches)):
        dist_kv_caches.append(torch.tensor(torch.cat((dist_key_caches[i].unsqueeze(0), dist_value_caches[i].unsqueeze(0)), dim=0)))

    swap_blocks(src_kv_caches[0], dist_kv_caches[0], block_mapping_tensor)


    for src, dst in block_mapping:
        torch.testing.assert_close(src_key_caches_clone[src].cpu(),
                                   dist_kv_caches[0][0][dst].cpu())
        torch.testing.assert_close(src_value_caches_clone[src].cpu(),
                                   dist_kv_caches[0][1][dst].cpu())
