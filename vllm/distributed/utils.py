# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import json
import os
from typing import Sequence

import torch
import torch.distributed as dist

from .parallel_state import get_cpu_world_group


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


_gpu_p2p_access_cache = None


def gpu_p2p_access_check(i: int, j: int) -> bool:
    """Check if GPU i can access GPU j."""

    # if the cache variable is already calculated,
    # read from the cache instead of checking it again
    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{i}->{j}"]

    is_distributed = dist.is_initialized()
    cpu_world_group = get_cpu_world_group()

    num_dev = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))
    path = os.path.expanduser(
        f"~/.config/vllm/gpu_p2p_access_cache_for_{cuda_visible_devices}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if (not is_distributed or dist.get_rank(cpu_world_group)== 0) \
        and (not os.path.exists(path)):
        # only the master process can enter this block to calculate the cache
        cache = {}
        for _i in range(num_dev):
            for _j in range(num_dev):
                cache[f"{_i}->{_j}"] = torch.cuda.can_device_access_peer(
                    _i, _j)
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
    if is_distributed and cpu_world_group is not None:
        dist.barrier(cpu_world_group)
    with open(path, "r") as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{i}->{j}"]
