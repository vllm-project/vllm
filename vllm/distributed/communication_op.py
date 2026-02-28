# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

import vllm.envs as envs

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _compute_balanced_split_sizes(total: int, world_size: int) -> list[int]:
    base = total // world_size
    remainder = total % world_size
    return [base + (1 if rank < remainder else 0) for rank in range(world_size)]


def _infer_sp_ragged_sizes_for_all_gather(
    input_: torch.Tensor,
    dim: int,
    tp_group: Any,
) -> list[int] | None:
    if not envs.VLLM_ENABLE_SP_RAGGED or dim != 0:
        return None

    # Avoid circular import at module import time.
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or not config.parallel_config.use_sequence_parallel_moe:
        return None

    world_size = tp_group.world_size
    if world_size <= 1:
        return None

    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return None

    if not is_forward_context_available():
        return None

    batch_descriptor = get_forward_context().batch_descriptor
    if batch_descriptor is None:
        return None

    total_tokens = batch_descriptor.num_tokens
    if total_tokens % world_size == 0:
        return None

    sizes = _compute_balanced_split_sizes(total_tokens, world_size)
    if input_.shape[dim] != sizes[tp_group.rank_in_group]:
        return None

    return sizes


def tensor_model_parallel_all_gather(
    input_: torch.Tensor,
    dim: int = -1,
    sizes: list[int] | None = None,
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    tp_group = get_tp_group()
    if sizes is None:
        normalized_dim = _normalize_dim(dim, input_.dim())
        sizes = _infer_sp_ragged_sizes_for_all_gather(input_, normalized_dim, tp_group)
    return tp_group.all_gather(input_, dim, sizes=sizes)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor,
    dim: int = -1,
    sizes: list[int] | None = None,
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim, sizes=sizes)


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
