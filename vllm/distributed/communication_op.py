from typing import Any, Dict, Optional, Union

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


def send_tensor_dict(tensors: Dict[str, torch.Tensor],
                     dst: Optional[int] = None) -> None:
    """
    Send the tensors to the next pipeline model parallel rank.
    Args:
        tensors (Dict[torch.Tensor]): Dict of tensors to send.
    """
    if dst is None:
        dst = get_pp_group().next_rank
    get_pp_group().send_tensor_dict(tensors, dst)


def recv_tensor_dict(
    src: Optional[int] = None
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """
    Receive tensors from the previous pipeline model parallel rank assuming all
    tensors are the same size.
    Returns:
        Dict[torch.Tensor]: Dict of received tensors.
    """
    if src is None:
        src = get_pp_group().prev_rank
    tensors = get_pp_group().recv_tensor_dict(src)
    return tensors
