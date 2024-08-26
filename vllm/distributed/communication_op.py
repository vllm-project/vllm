from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


@torch.library.custom_op("vllm::tp_out_of_place_ar",
                         mutates_args=["input_"],
                         device_types=("cuda", "cpu"))
def _tp_out_of_place_ar(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().out_of_place_ar(input_)


@torch.library.register_fake("vllm::tp_out_of_place_ar")
def _tp_out_of_place_ar_fake(input_: torch.Tensor) -> torch.Tensor:
    return input_


@torch.library.custom_op("vllm::tp_in_place_ar",
                         mutates_args=["input_"],
                         device_types=("cuda", "cpu"))
def _tp_in_place_ar(input_: torch.Tensor) -> None:
    """All-reduce the input tensor across model parallel group."""
    get_tp_group().in_place_ar(input_)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if get_tp_group().should_run_out_of_place_ar(input_):
        return torch.ops.vllm.tp_out_of_place_ar(input_)
    else:
        torch.ops.vllm.tp_in_place_ar(input_)
        return input_


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
