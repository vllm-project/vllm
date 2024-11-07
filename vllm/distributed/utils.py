# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from typing import Sequence, Tuple

import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                is_nccl_available)
from torch.distributed.rendezvous import rendezvous

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


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


def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                   pp_size: int) -> Tuple[int, int]:
    """Try to evenly distribute layers across partitions.
    If the number of layers is not divisible by the number of partitions,
    the last partition will have the remaining layers.
    """
    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    if partition_list_str is not None:
        try:
            partitions = [
                int(layer) for layer in partition_list_str.split(",")
            ]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(
                partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(
                f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
    else:
        layers_per_partition = num_hidden_layers // pp_size
        start_layer = pp_rank * layers_per_partition
        end_layer = start_layer + layers_per_partition

        if pp_rank == pp_size - 1:
            end_layer = num_hidden_layers

    return (start_layer, end_layer)


def stateless_init_process_group(init_method: str, rank: int, world_size: int,
                                 backend: str) -> ProcessGroup:
    """A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state.

    If we have process A and process B called `torch.distributed.init_process_group`
    to form a group, and then we want to form another group with process A, B, C,
    D, it is not possible in PyTorch, because process A and process B have already
    formed a group, and process C and process D cannot join that group. This
    function is a workaround for this issue.

    `torch.distributed.init_process_group` is a global call, while this function
    is a stateless call. It will return a `ProcessGroup` object that can be used
    for collective communication. With this function, process A and process B
    can call `stateless_init_process_group` to form a group, and then process A, B,
    C, and D can call `stateless_init_process_group` to form another group.
    """ # noqa

    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    pg_options = ProcessGroup.Options(backend=backend, timeout=timeout)

    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
        pg_options,
    )

    if backend == "gloo":
        from torch.distributed.distributed_c10d import ProcessGroupGloo
        backend_class = ProcessGroupGloo(prefix_store,
                                         group_rank,
                                         group_size,
                                         timeout=timeout)
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
    elif backend == "nccl":
        assert is_nccl_available()
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")

    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)

    return pg
