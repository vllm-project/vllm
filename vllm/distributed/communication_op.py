from typing import Any, Dict, Optional, Union

import torch

from .parallel_state import get_tp


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


def broadcast_tensor_dict(self,
                          tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    return get_tp().broadcast_tensor_dict(tensor_dict, src)
