from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup

from .parallel_state import get_cpu_world_group, get_tp


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    return get_tp().gather(input_, dst, dim)


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
            if tensor.numel() == 0:
                # Skip broadcasting empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                handle = torch.distributed.broadcast(tensor,
                                                     src=src,
                                                     group=metadata_group,
                                                     async_op=True)
            else:
                # use group for GPU tensors
                handle = torch.distributed.broadcast(tensor,
                                                     src=src,
                                                     group=group,
                                                     async_op=True)
            async_handles.append(handle)
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
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
