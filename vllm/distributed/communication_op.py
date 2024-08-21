from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group

torch.library.define("vllm::tensor_model_parallel_all_reduce",
                     ("(Tensor(a!) input_ ) -> Tensor"))


@torch.library.register_kernel("vllm::tensor_model_parallel_all_reduce",
                               ("cuda", "cpu"))
def _tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().out_of_place_ar(input_)


@torch.library.register_fake("vllm::tensor_model_parallel_all_reduce")
def _tensor_model_parallel_all_reduce_fake(
        input_: torch.Tensor) -> torch.Tensor:
    return input_


torch.library.define("vllm::tensor_model_parallel_all_reduce_in_place",
                     ("(Tensor! input_ ) -> ()"))


@torch.library.register_kernel(
    "vllm::tensor_model_parallel_all_reduce_in_place", ("cuda", "cpu"))
def _tensor_model_parallel_all_reduce_in_place(input_: torch.Tensor):
    """All-reduce the input tensor across model parallel group."""
    get_tp_group().in_place_ar(input_)


@torch.library.register_fake("vllm::tensor_model_parallel_all_reduce_in_place")
def _tensor_model_parallel_all_reduce__in_place_fake(input_: torch.Tensor):
    return


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if get_tp_group().should_run_in_place_ar(input_):
        torch.ops.vllm.tensor_model_parallel_all_reduce_in_place(input_)
        return input_
    else:
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
