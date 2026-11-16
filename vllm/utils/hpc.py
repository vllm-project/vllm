# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for HPC API changes.

Users of vLLM should always import **only** these wrappers.
"""

import functools
import importlib
import importlib.util
import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_hpc() -> bool:
    """Return `True` if hpc package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("hpc") is None:
        logger.warning_once(
            "HPC attention requires the hpc module to be installed. "
            "Please install it from https://github.com/Tencent/hpc-ops"
        )
        return False
    return True


# Remove 'torch._library.custom_ops':
# The output of this custom operator (1) must not also be an input to
# this custom operator and (2) may not alias any inputs to this custom
# operator or other returns. The most common way to trigger this error
# is if we have y = custom_op(x) and y and x are the same Tensor.
# Please instead return a clone of the offending output tensor(s) (e.g.
# return x.clone()) or refactor the custom operator to not return y.
# @torch.library.custom_op(
#     "vllm::fuse_moe_impl",
#     mutates_args=[],
#     device_types="cuda",
# )
def fuse_moe_impl(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    act_and_mul_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    from hpc import fuse_moe as fuse_moe_

    return fuse_moe_(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        shared_output,
        output=output,
    )


def hpc_fuse_moe(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    act_and_mul_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    return fuse_moe_impl(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        shared_output,
        output=output,
    )


# @torch.library.custom_op(
#     "vllm::fuse_moe_blockwise_impl",
#     mutates_args=[],
#     device_types="cuda",
# )
def fuse_moe_blockwise_impl(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    from hpc import fuse_moe_blockwise as fuse_moe_blockwise_

    return fuse_moe_blockwise_(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output=output,
    )


def hpc_fuse_moe_blockwise(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: torch.Tensor = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    return fuse_moe_blockwise_impl(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output=output,
    )


@functools.cache
def has_hpc_sm120_moe() -> bool:
    """Return `True` if the installed HPC-Ops has the SM120 fused MoE."""
    if not has_hpc():
        return False
    try:
        from hpc import fuse_moe_mxfp8  # noqa: F401
    except ImportError:
        logger.warning_once(
            "The installed hpc package has no SM120 fused MoE; it needs "
            "a newer https://github.com/Tencent/hpc-ops built for SM120."
        )
        return False
    return True


def hpc_fuse_moe_mxfp8_sm120(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    output: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """MXFP8 or MXFP4 weight x MXFP8 activation fused MoE on SM120.

    `x` is MXFP8 with its UE8M0 `x_scale`, or BF16 with `x_scale=None`, which
    is quantized to MXFP8 here. Weights and scales come from
    `prepare_mx_moe_layer_for_hpc_sm120` in `fused_moe.hpc_moe`.
    """
    from hpc import fuse_moe_mxfp8

    if x_scale is None:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        x, x_scale = mxfp8_e4m3_quantize(x)
        x_scale = x_scale.view(torch.uint8)

    return fuse_moe_mxfp8(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        limit=math.inf if swiglu_limit is None else swiglu_limit,
        output=output,
    )


__all__ = [
    "has_hpc",
    "has_hpc_sm120_moe",
    "hpc_fuse_moe",
    "hpc_fuse_moe_blockwise",
    "hpc_fuse_moe_mxfp8_sm120",
]
