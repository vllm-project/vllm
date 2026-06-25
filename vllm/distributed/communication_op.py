# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def _all_reduce_with_dbo_yields(input_: torch.Tensor) -> torch.Tensor:
    try:
        from vllm.v1.worker.ubatching import (
            dbo_enabled,
            dbo_yield_and_switch_from_comm_to_compute,
            dbo_yield_and_switch_from_compute_to_comm,
        )
    except Exception:
        return get_tp_group().all_reduce(input_)
    if not dbo_enabled():
        return get_tp_group().all_reduce(input_)
    dbo_yield_and_switch_from_compute_to_comm()
    out = get_tp_group().all_reduce(input_)
    dbo_yield_and_switch_from_comm_to_compute()
    return out


# Opaque custom op so torch.compile can't fold the runtime dbo_enabled() check.
try:
    from vllm.utils.torch_utils import direct_register_custom_op

    def _ar_op_impl(input_: torch.Tensor) -> torch.Tensor:
        return _all_reduce_with_dbo_yields(input_)

    def _ar_op_fake(input_: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(input_)

    direct_register_custom_op(
        op_name="vllm_dbo_all_reduce",
        op_func=_ar_op_impl,
        mutates_args=[],
        fake_impl=_ar_op_fake,
    )
    _AR_OP = torch.ops.vllm.vllm_dbo_all_reduce.default
except Exception:
    _AR_OP = None


# Route AR through the opaque op only when DBO is configured (resolved once when
# the engine config is available; a compile-time constant for that engine).
_USE_DBO_AR_OP: bool | None = None


def _use_dbo_all_reduce_op() -> bool:
    global _USE_DBO_AR_OP
    if _USE_DBO_AR_OP is None and _AR_OP is not None:
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            _USE_DBO_AR_OP = cfg.parallel_config.use_ubatching
    return bool(_USE_DBO_AR_OP)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if _use_dbo_all_reduce_op():
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
