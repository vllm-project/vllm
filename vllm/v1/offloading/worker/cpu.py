# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.attention import AttentionBackend
from vllm.utils import is_pin_memory_available
from vllm.v1.offloading.abstract import LoadStoreSpec
from vllm.v1.offloading.mediums import BlockIDLoadStoreSpec
from vllm.v1.offloading.worker.worker import TransferFunction, TransferSpec


def create_cpu_tensors(
    gpu_kv_caches: dict[str, torch.Tensor],
    gpu_block_size: int,
    cpu_block_size: int,
    num_cpu_blocks: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Create tensors for the CPU KV cache.

    Args:
        gpu_kv_caches: The per-layer GPU KV cache tensors
        gpu_block_size: Number of tokens per GPU block
        cpu_block_size: Number of tokens per CPU block
        num_cpu_blocks: The number of CPU blocks to allocate

    Note:
        - The GPU block size must divide the CPU block size.
        - The shape of the GPU KV cache must be (2, num_blocks, ...)

    Returns:
        Matching per-layer lists of (gpu_tensors, cpu_tensors).
    """
    assert cpu_block_size % gpu_block_size == 0

    pin_memory = is_pin_memory_available()

    gpu_tensors = []
    cpu_tensors = []
    for gpu_tensor in gpu_kv_caches.values():
        gpu_shape = gpu_tensor.shape
        assert len(gpu_shape) >= 4  # (2, num_blocks, ..., ...)
        assert gpu_shape[0] == 2

        cpu_shape = list(gpu_shape)
        cpu_shape[1] = num_cpu_blocks * (cpu_block_size // gpu_block_size)

        gpu_tensors.append(gpu_tensor)
        cpu_tensors.append(
            torch.zeros(cpu_shape,
                        dtype=gpu_tensor.dtype,
                        device="cpu",
                        pin_memory=pin_memory))

    return gpu_tensors, cpu_tensors


def block_ids(specs_list: list[LoadStoreSpec],
              block_size_factor: int,
              skip_count: int = 0) -> Iterator[int]:
    """
    Convert a list of BlockIDLoadStoreSpec to a list of matching block ids,
    assuming each spec is composed of actual block_size_factor blocks.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if spec_list = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    for spec in specs_list:
        assert isinstance(spec, BlockIDLoadStoreSpec)
        base_block_id = spec.block_id * block_size_factor
        for i in range(skip_count, block_size_factor):
            yield base_block_id + i

        # finished skipping
        skip_count = 0


def generate_tensors_transfer_function(
    src_tensors: list[torch.Tensor],
    dst_tensors: list[torch.Tensor],
    attn_backend: type[AttentionBackend],
    src_block_size: int,
    dst_block_size: int,
) -> TransferFunction:
    """
    Generate a function for transferring from one KV cache to another.

    Args:
        src_tensors: the per-layer tensors of the source KV cache.
        dst_tensors: the per-layer tensors of the destination KV cache.
        attn_backend: the attention backend for both caches.
        src_block_size: the block size of the source KV cache.
        dst_block_size: the block size of the destination KV cache.

    Returns:
        A function for executing transfers between the caches.

    Note: one of src_block_size, dst_block_size must divide the other.
    """
    assert len(src_tensors) == len(dst_tensors)

    min_block_size = min(src_block_size, dst_block_size)
    max_block_size = max(src_block_size, dst_block_size)
    assert max_block_size % min_block_size == 0

    src_block_size_factor = src_block_size // min_block_size
    dst_block_size_factor = dst_block_size // min_block_size

    def transfer_function(spec: TransferSpec) -> bool:
        src_blocks_specs_list, dst_blocks_specs_list = spec

        dst_sub_blocks_to_skip = (-len(src_blocks_specs_list) %
                                  dst_block_size_factor)

        assert (len(src_blocks_specs_list) *
                src_block_size_factor == len(dst_blocks_specs_list) *
                dst_block_size_factor - dst_sub_blocks_to_skip)

        src_to_dst_list: list[tuple[int, int]] = list(
            zip(
                block_ids(src_blocks_specs_list, src_block_size_factor),
                block_ids(dst_blocks_specs_list, dst_block_size_factor,
                          dst_sub_blocks_to_skip)))
        src_to_dst = torch.tensor(src_to_dst_list,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)

        # iterate over layers
        for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
            attn_backend.swap_blocks(src_tensor, dst_tensor, src_to_dst)

        # always successful
        return True

    return transfer_function
