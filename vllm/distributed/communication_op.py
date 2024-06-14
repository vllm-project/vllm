from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_pp_group, get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def send_next_rank(tensors: List[torch.Tensor]) -> None:
    """
    Send the tensors to the next pipeline model parallel rank.
    Args:
        tensors (List[torch.Tensor]): List of tensors to send.
    """
    next_rank = get_pp_group().next_rank
    for tensor in tensors:
        get_pp_group().send(tensor, dst=next_rank)


def recv_prev_rank(num_tensors: int, size: torch.Size,
                   dtype: torch.dtype) -> List[torch.Tensor]:
    """
    Receive tensors from the previous pipeline model parallel rank assuming all
    tensors are the same size.
    Args:
        num_tensors (int): Number of tensors to receive.
        size (torch.Size): Size of the tensors.
        dtype (torch.dtype): Data type of the tensors.
    Returns:
        List[torch.Tensor]: List of received tensors.
    """
    prev_rank = get_pp_group().prev_rank
    tensors = []
    for _ in range(num_tensors):
        tensor = get_pp_group().recv(size, dtype, src=prev_rank)
        tensors.append(tensor)
    return tensors
