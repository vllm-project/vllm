from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group

torch.library.define("vllm::tensor_model_parallel_all_reduce",
                     ("(Tensor(a!) input_ ) -> Tensor(a)"))


@torch.library.register_kernel("vllm::tensor_model_parallel_all_reduce", ("cuda", "cpu"))
def _tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


@torch.library.register_fake("vllm::tensor_model_parallel_all_reduce")
def _tensor_model_parallel_all_reduce_fake(
        input_: torch.Tensor) -> torch.Tensor:
    return input_


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.tensor_model_parallel_all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
