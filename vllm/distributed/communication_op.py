# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def _custom_sequence_parallel_collective(
    name: str, input_: torch.Tensor
) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(input_)


def sequence_parallel_all_gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather token shards across the tensor-parallel group."""
    output = _custom_sequence_parallel_collective("custom_all_gather", input_)
    if output is None:
        output = tensor_model_parallel_all_gather(input_, dim=0)
    return output


def sequence_parallel_reduce_scatter(input_: torch.Tensor) -> torch.Tensor:
    """Sum partial results and scatter them along the token dimension."""
    output = _custom_sequence_parallel_collective("custom_reduce_scatter", input_)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(input_, dim=0)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
