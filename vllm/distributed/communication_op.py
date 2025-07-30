# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import torch
import torch.distributed
from contextlib import nullcontext
from vllm.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory, dummy_symmetric_memory)
from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor, output_: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_, output_)


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


def tensor_model_parallel_use_symmetric_memory():
    # TODO(asamani): remove this once we have a way to torch compile with 
    # mempool or break the graph.
    if torch.compiler.is_compiling():
        return dummy_symmetric_memory()
    return use_symmetric_memory(get_tp_group())

def tensor_model_parallel_maybe_get_symm_buffer(shape, dtype):
    # in case symm memory is available for this process group, we return a
    # tensor in symm memory, otherwise we return a regular torch tensor.
    return get_tp_group().maybe_get_symm_buffer(shape, dtype)
