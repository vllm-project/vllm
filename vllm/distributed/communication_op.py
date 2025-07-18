# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import torch
import torch.distributed
from contextlib import nullcontext
from vllm.distributed.device_communicators.pynccl_allocator import (
    get_nccl_mem_pool)
from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(input_: torch.Tensor,
                                         dim: int = -1) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)

def tensor_model_parallel_mempool_ctx():
    # TODO(asamani): check arg for enabling this feature
    return torch.cuda.use_mem_pool(get_nccl_mem_pool())


def tensor_model_parallel_register_window(tensor: torch.Tensor):
    return get_tp_group().pynccl_comm.register_comm_window(tensor)


def tensor_model_parallel_deregister_window(window):
    get_tp_group().pynccl_comm.deregister_comm_window(window)

