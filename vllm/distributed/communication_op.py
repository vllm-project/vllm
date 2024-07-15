from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup
from vllm._C import ops

from .parallel_state import (get_cpu_world_group, get_pp_pynccl_communicator,
                             get_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size,
                             get_tp_ca_communicator,
                             get_tp_pynccl_communicator)


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


@contextmanager
def graph_capture():
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    stream = torch.cuda.Stream()
    graph_capture_context = GraphCaptureContext(stream)
    ca_comm = get_tp_ca_communicator()
    maybe_ca_context = nullcontext() if ca_comm is None else ca_comm.capture()
    with torch.cuda.stream(stream), maybe_ca_context:
        # In graph mode, we have to be very careful about the collective
        # operations. The current status is:
        #     allreduce \ Mode   |  Eager  |  Graph  |
        # --------------------------------------------
        # custom allreduce       | enabled | enabled |
        # PyNccl                 | disabled| enabled |
        # torch.distributed      | enabled | disabled|
        #
        # Note that custom allreduce will have a runtime check, if the tensor
        #  size is too large, it will fallback to the next available option.
        # In summary: When using CUDA graph, we use
        # either custom all-reduce kernel or pynccl. When not using CUDA
        # graph, we use either custom all-reduce kernel or PyTorch NCCL.
        # We always prioritize using custom all-reduce kernel but fall back
        # to PyTorch or pynccl if it is disabled or not supported.
        tp_pynccl_comm = get_tp_pynccl_communicator()
        pp_pynccl_comm = get_pp_pynccl_communicator()
        if not tp_pynccl_comm:
            maybe_tp_pynccl_context = nullcontext()
        else:
            maybe_tp_pynccl_context = tp_pynccl_comm.change_state(
                enable=True, stream=torch.cuda.current_stream())
        if not pp_pynccl_comm:
            maybe_pp_pynccl_context = nullcontext()
        else:
            maybe_pp_pynccl_context = pp_pynccl_comm.change_state(
                enable=True, stream=torch.cuda.current_stream())
        with maybe_tp_pynccl_context, maybe_pp_pynccl_context:
            yield graph_capture_context


@torch.library.impl("myops::_tensor_model_parallel_all_reduce", "cpu")
def _tensor_model_parallel_all_reduce(
    input_: torch.Tensor):
    ops.shm_allreduce(input_, get_tensor_model_parallel_rank())
    return input_

torch.library.define(
    "myops::_tensor_model_parallel_all_reduce",
    "(Tensor input_) -> Tensor",
)

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation will be applied in-place on the input tensor if
    disable_custom_all_reduce is set to True. Otherwise, this operation may or
    may not be applied in place depending on whether custom all reduce is
    invoked for a particular tensor, which further depends on the tensor size
    and GPU topology.

    TLDR: always assume this function modifies its input, but use the return
    value as the output.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    return torch.ops.myops._tensor_model_parallel_all_reduce(input_)


@torch.library.impl("myops::_tensor_model_parallel_all_gather", "cpu")
def _tensor_model_parallel_all_gather(
    input_: torch.Tensor, world_size:int, dim: int = -1):
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor

torch.library.define(
    "myops::_tensor_model_parallel_all_gather",
    "(Tensor input_, int world_size, int dim) -> Tensor",
)

def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    return torch.ops.myops._tensor_model_parallel_all_gather(input_, world_size, dim)

@torch.library.impl("myops::_tensor_model_parallel_gather", "cpu")
def _tensor_model_parallel_gather(
    input_: torch.Tensor, world_size:int, dst: int = 0, dim: int = -1):
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    if get_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst,
                             group=get_tensor_model_parallel_group())
    if get_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


torch.library.define(
    "myops::_tensor_model_parallel_gather",
    "(Tensor input_, int world_size, int dst, int dim) -> Tensor",
)

def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    return torch.ops.myops._tensor_model_parallel_gather(input_, world_size, dst, dim)




def broadcast(input_: torch.Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None):
    """Broadcast the input tensor."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_


def broadcast_object_list(obj_list: List[Any],
                          src: int = 0,
                          group: Optional[ProcessGroup] = None):
    """Broadcast the input object list."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
    tensor_dict: Dict[Any, Union[torch.Tensor, Any]]
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list = []
    tensor_list = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = "cpu" if value.is_cpu else "cuda"
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size())))
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary.
    `group` is used to broadcast the tensors, while `metadata_group` is used
     to broadcast the metadata of the dict (e.g. dict structure, tensor sizes,
     dtypes).
    """
    # Bypass the function if we are using only 1 GPU.
    if (not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(group=group) == 1):
        return tensor_dict

    group = group or torch.distributed.group.WORLD
    metadata_group = metadata_group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict

    broadcast_list = [tensor_dict]
    torch.distributed.broadcast_object_list(broadcast_list,
                                            src=src,
                                            group=group)
    return broadcast_list[0]
