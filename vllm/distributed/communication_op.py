from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup

from .parallel_state import (get_cpu_world_group,
                             get_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size,
                             is_pynccl_enabled_for_all_reduce)


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
    from vllm.distributed.device_communicators import pynccl_utils
    from vllm.distributed.device_communicators.custom_all_reduce import (
        custom_all_reduce)

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    out = custom_all_reduce(input_)
    if out is not None:
        return out
    if is_pynccl_enabled_for_all_reduce():
        pynccl_utils.all_reduce(input_)
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


TensorMetadata = namedtuple("TensorMetadata", ["dtype", "size"])


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
            # Note(youkaichao): currently this only supports broadcasting
            # tensors on cuda. In the future, we can add device as a field in
            # TensorMetadata to support broadcasting tensors on different
            # devices.
            assert value.is_cuda, (
                f"Tensor {key}: {value} is not on cuda. Currently we only "
                f"support broadcasting tensors on cuda.")
            metadata_list.append((key, TensorMetadata(value.dtype,
                                                      value.size())))
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
    group = group or torch.distributed.group.WORLD
    metadata_group = metadata_group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict

    rank = torch.distributed.get_rank()
    if rank == src:
        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `broadcast_object_list` involves serialization and deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        torch.distributed.broadcast_object_list([metadata_list],
                                                src=src,
                                                group=metadata_group)
        async_handles = []
        for tensor in tensor_list:
            async_handles.append(
                torch.distributed.broadcast(tensor,
                                            src=src,
                                            group=group,
                                            async_op=True))
        for async_handle in async_handles:
            async_handle.wait()

    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list,
                                                src=src,
                                                group=metadata_group)
        assert recv_metadata_list[0] is not None
        tensor_dict = {}
        async_handles = []
        for key, value in recv_metadata_list[0]:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device="cuda")
                async_handle = torch.distributed.broadcast(tensor,
                                                           src=src,
                                                           async_op=True,
                                                           group=group)
                async_handles.append(async_handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
