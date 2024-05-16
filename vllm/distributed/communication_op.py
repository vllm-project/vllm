import pickle
import struct
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .parallel_state import (get_cpu_world_group,
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
        pynccl_comm = get_tp_pynccl_communicator()
        if pynccl_comm is None:
            maybe_pynccl_context = nullcontext()
        else:
            maybe_pynccl_context = pynccl_comm.change_state(
                enable=True, stream=torch.cuda.current_stream())
        with maybe_pynccl_context:
            yield graph_capture_context


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
    ca_comm = get_tp_ca_communicator()

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    if ca_comm is not None:
        out = ca_comm.custom_all_reduce(input_)
        if out is not None:
            return out
    pynccl_comm = get_tp_pynccl_communicator()
    if (pynccl_comm is not None and not pynccl_comm.disabled):
        pynccl_comm.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_,
                                     group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
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
    tensor_dict: Dict[str, Union[torch.Tensor, Any]]
) -> Tuple[List[Any], List[torch.Tensor]]:
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


def _create_buffer(size: int) -> Tuple[memoryview, torch.Tensor]:
    buffer = memoryview(bytearray(size))
    return buffer, torch.frombuffer(buffer, dtype=torch.uint8)


RESIZE_MESSAGE_FLAG = 1

# Globals
CALLSITE_TENSOR_SIZES = [16]
BROADCAST_BUFFER, BROADCAST_TENSOR = _create_buffer(16)


def register_broadcast_callsite() -> int:
    """Obtain a unique call site id for use with broadcast_tensor_dict"""
    global CALLSITE_TENSOR_SIZES
    CALLSITE_TENSOR_SIZES.append(16)
    return len(CALLSITE_TENSOR_SIZES) - 1


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None,
    callsite_id: int = 0,
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary.
    `group` is used to broadcast the tensors, while `metadata_group` is used
     to broadcast the metadata of the dict (e.g. dict structure, tensor sizes,
     dtypes).
     `callsite_id` should be obtained for a particular call site by
     calling register_broadcast_callsite()
    """
    group = group or torch.distributed.group.WORLD

    # Bypass the function if we are using only 1 GPU.
    if (not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(group=group) == 1):
        return tensor_dict

    metadata_group = metadata_group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    global CALLSITE_TENSOR_SIZES, BROADCAST_BUFFER, BROADCAST_TENSOR
    callsite_tensor_size = CALLSITE_TENSOR_SIZES[callsite_id]

    rank = torch.distributed.get_rank()
    if rank == src:
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        pickled = pickle.dumps(metadata_list)
        size = len(pickled) + 8

        if callsite_tensor_size < size:
            # Increase call site tensor size. First byte == 1 means resize.
            BROADCAST_BUFFER[0] = RESIZE_MESSAGE_FLAG
            BROADCAST_BUFFER[8:16] = struct.pack('<Q', size)
            dist.broadcast(BROADCAST_TENSOR[:callsite_tensor_size],
                           src=src,
                           group=metadata_group)
            callsite_tensor_size = size
            # Grow buffer if needed.
            if len(BROADCAST_BUFFER) < callsite_tensor_size:
                BROADCAST_BUFFER, BROADCAST_TENSOR = _create_buffer(
                    callsite_tensor_size)
            else:
                BROADCAST_BUFFER[0] = 0
            CALLSITE_TENSOR_SIZES[callsite_id] = callsite_tensor_size

        # Copy pickled metadata into buffer and broadcast it.
        BROADCAST_BUFFER[8:size] = pickled
        dist.broadcast(BROADCAST_TENSOR[:callsite_tensor_size],
                       src=src,
                       group=metadata_group)

        async_handles = []
        for tensor in tensor_list:
            # Skip broadcasting empty tensors.
            if tensor.numel() != 0:
                # use metadata_group for CPU tensors, group for GPU tensors.
                handle = dist.broadcast(
                    tensor,
                    src=src,
                    group=metadata_group if tensor.is_cpu else group,
                    async_op=True)
                async_handles.append(handle)
        for async_handle in async_handles:
            async_handle.wait()

    else:
        while True:
            dist.broadcast(BROADCAST_TENSOR[:callsite_tensor_size],
                           src=src,
                           group=metadata_group)
            if not BROADCAST_BUFFER[0]:
                break
            # Increase call site tensor size.
            callsite_tensor_size, = struct.unpack('<Q', BROADCAST_BUFFER[8:16])
            # Grow buffer if needed.
            if len(BROADCAST_BUFFER) < callsite_tensor_size:
                BROADCAST_BUFFER, BROADCAST_TENSOR = _create_buffer(
                    callsite_tensor_size)
            CALLSITE_TENSOR_SIZES[callsite_id] = callsite_tensor_size
            # Repeat broadcast.

        recv_metadata = pickle.loads(BROADCAST_BUFFER[8:callsite_tensor_size])

        tensor_dict = {}
        async_handles = []
        for key, value in recv_metadata:
            if not isinstance(value, TensorMetadata):
                tensor_dict[key] = value
                continue

            tensor = torch.empty(value.size,
                                 dtype=value.dtype,
                                 device=value.device)
            # Skip broadcasting empty tensors.
            if tensor.numel() != 0:
                # use metadata_group for CPU tensors, group for GPU tensors.
                handle = dist.broadcast(
                    tensor,
                    src=src,
                    group=metadata_group if tensor.is_cpu else group,
                    async_op=True)
                async_handles.append(handle)

            tensor_dict[key] = tensor

        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
