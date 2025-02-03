# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import pickle
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import torch
from torch.distributed import TCPStore

import vllm.envs as envs
from vllm.logger import init_logger
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
import torch_xla.core.xla_model as xm
import torch_xla

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
    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: Dict[int, int] = dataclasses.field(
        default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: Deque[Tuple[str,
                         float]] = dataclasses.field(default_factory=deque)

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
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def expire_data(self):
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.time() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(
                f"send_to/{self.rank}/{self.recv_src_counter[src]}"))
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = (f"broadcast_from/{src}/"
                   f"{self.broadcast_send_counter}")
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return obj
        else:
            key = (f"broadcast_from/{src}/"
                   f"{self.broadcast_recv_src_counter[src]}")
            recv_obj = pickle.loads(self.store.get(key))
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

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
    ) -> "StatelessProcessGroup":
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
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """ # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds)


def initialize_spmd():
    global mesh, device_ids
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh
    import numpy as np

    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, )
    logger.info(f"hosseins: mesh_shape: [{mesh_shape=}]")
    device_ids = np.array(range(num_devices))
    _mesh = Mesh(device_ids, mesh_shape, ('axis', ))
    mesh = _mesh
    return _mesh

def get_mesh():
    # return None
    global mesh
    if mesh is None:
        logger.info('hosseins: creating mesh')
        mesh = initialize_spmd()
    else:
        logger.info('hosseins: returning mesh')
        return mesh

mesh = None
device_ids = None

# def get_col_parallel_partition_spec():
#     return ('axis', None)
#     # return ('data', 'model')
# 
# def get_row_parallel_partition_spec():
#     return (None, 'axis')
#     # return ('model', 'data')

def get_col_parallel_partition_spec():
    # return ('axis', None)
    return (None, 'axis')

def get_row_parallel_partition_spec():
    # return (None, 'axis')
    return ('axis', None)

def shard_spmd(data, mesh, partition_spec, show_visual=False):
    assert isinstance(data, torch.Tensor), "Object is not an torch.Tensor"
    # xs.mark_sharding(data, mesh, partition_spec)
    xm.mark_step()
    logger.info(f"hosseins: shard_spmd() -> [{type(data)=}]")
    # sharding = torch_xla._XLAC._get_xla_sharding_spec(data)
    # logger.info(f"hosseins: shard_spmd() -> [{sharding=}]")

    if show_visual:
        logger.info("hosseins: after sharding param")
        generated_table = visualize_tensor_sharding(data, use_color=False)


def shard_spmd(data, mesh, partition_spec, show_visual=False):
    assert isinstance(data, torch.Tensor), "Object is not an torch.Tensor"
    # mesh_shape = (len(device_ids), )
    # axis_names = "('axis', )" # string version of axis_names
    # # partition_spec = "('data', 'model')" # string version of partition spec
    # torch.ops.xla.dynamo_mark_sharding(data, device_ids, mesh_shape, axis_names, partition_spec)

    xs.mark_sharding(data, mesh, partition_spec)
    xm.mark_step()
    logger.info(f"hosseins: shard_spmd() -> [{type(data)=}]")
    # sharding = torch_xla._XLAC._get_xla_sharding_spec(data)
    # logger.info(f"hosseins: shard_spmd() -> [{sharding=}]")

    if show_visual:
        logger.info("hosseins: after sharding param")
        generated_table = visualize_tensor_sharding(data, use_color=False)
