# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.utils.torch_utils import direct_register_custom_op

from .parallel_state import get_tp_group


def _all_reduce_with_dbo_yields(input_: torch.Tensor) -> torch.Tensor:
    from vllm.v1.worker.ubatching import (
        dbo_enabled,
        dbo_yield_and_switch_from_comm_to_compute,
        dbo_yield_and_switch_from_compute_to_comm,
    )

    if not dbo_enabled():
        return get_tp_group().all_reduce(input_)
    dbo_yield_and_switch_from_compute_to_comm()
    out = get_tp_group().all_reduce(input_)
    dbo_yield_and_switch_from_comm_to_compute()
    return out


# Opaque custom op so torch.compile can't fold the runtime dbo_enabled() check.
def _ar_op_fake(input_: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input_)


direct_register_custom_op(
    op_name="vllm_dbo_all_reduce",
    op_func=_all_reduce_with_dbo_yields,
    mutates_args=[],
    fake_impl=_ar_op_fake,
)
_AR_OP = torch.ops.vllm.vllm_dbo_all_reduce.default


# Route AR through the opaque op only when DBO is configured. Resolved once at
# worker/model-runner init (while the engine config is in scope) and cached; it
# must NOT be read from get_current_vllm_config_or_none() at forward/compile
# time, where the config global is unset and would gate this to False.
_USE_DBO_AR_OP: bool = False


def configure_dbo_all_reduce(use_ubatching: bool) -> None:
    """Set whether TP all-reduce routes through the opaque DBO op.

    Call this during worker/model-runner initialization (config in scope),
    before model compilation/warmup runs the first forward.
    """
    global _USE_DBO_AR_OP
    _USE_DBO_AR_OP = bool(use_ubatching) and _AR_OP is not None


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if _USE_DBO_AR_OP:
        return _AR_OP(input_)
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
