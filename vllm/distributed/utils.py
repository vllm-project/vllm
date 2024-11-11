# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
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


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """
    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store
    prefix: str

    # dst rank -> counter
    send_dst_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: Dict[int, int] = dataclasses.field(
        default_factory=dict)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {
            i: 0
            for i in range(self.world_size)
        }

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.store.set(
            f"{self.prefix}/send_to/{dst}/{self.send_dst_counter[dst]}",
            pickle.dumps(obj))
        self.send_dst_counter[dst] += 1

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(
                f"{self.prefix}/send_to/{self.rank}/{self.recv_src_counter[src]}"
            ))
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.store.set(
                f"{self.prefix}/broadcast/{src}/{self.broadcast_send_counter}",
                pickle.dumps(obj))
            self.broadcast_send_counter += 1
            return obj
        else:
            recv_obj = pickle.loads(
                self.store.get(
                    f"{self.prefix}/broadcast/{src}/{self.broadcast_recv_src_counter[src]}"
                ))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def barrier(self):
        """A barrier to synchronize all ranks."""
        for i in range(self.world_size):
            if i == self.rank:
                self.broadcast_obj(None, src=self.rank)
            else:
                self.broadcast_obj(None, src=i)


def stateless_init_process_group(init_method: str, rank: int,
                                 world_size: int) -> StatelessProcessGroup:
    """A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state.

    If we have process A and process B called `torch.distributed.init_process_group`
    to form a group, and then we want to form another group with process A, B, C,
    D, it is not possible in PyTorch, because process A and process B have already
    formed a group, and process C and process D cannot join that group. This
    function is a workaround for this issue.

    `torch.distributed.init_process_group` is a global call, while this function
    is a stateless call. It will return a `StatelessProcessGroup` object that can be
    used for exchanging metadata. With this function, process A and process B
    can call `stateless_init_process_group` to form a group, and then process A, B,
    C, and D can call `stateless_init_process_group` to form another group.
    """ # noqa

    from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
    timeout = _DEFAULT_PG_TIMEOUT

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    return StatelessProcessGroup(rank, world_size, store, init_method)
