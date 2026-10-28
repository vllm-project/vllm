# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for HPC API changes.

Users of vLLM should always import **only** these wrappers.
"""

import functools
import importlib
import importlib.util

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
        from hpc import (  # noqa: F401
            fuse_moe_blockwise_fp8_sm120,
            fuse_moe_blockwise_mxfp4_sm120,
        )
    except ImportError:
        logger.warning_once(
            "The installed hpc package has no SM120 fused MoE; it needs "
            "a newer https://github.com/Tencent/hpc-ops built for SM120."
        )
        return False
    return True


def hpc_prepare_fp8_moe_weights_sm120(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize 128-block FP8 expert weights against UE8M0 scales in place
    and return them with the scales packed for the SM120 MoE."""
    from hpc import prepare_fp8_moe_weights_sm120

    return weight, prepare_fp8_moe_weights_sm120(weight, weight_scale)


def hpc_prepare_mxfp4_moe_weights_sm120(
    scale: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Prepare MXFP4 MoE weight scales for the SM120 MXFP4 fused MoE.

    Packs `[experts, rows, cols // 32]` UE8M0 scales into the GEMM layout.
    """
    from hpc import prepare_mxfp4_moe_weights_sm120

    return prepare_mxfp4_moe_weights_sm120(scale, rows, cols)


def hpc_fuse_moe_blockwise_fp8_sm120(
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
    """128-block FP8 fused MoE on SM120.

    `x` is BF16 with `x_scale=None`, which the kernel quantizes itself, or FP8
    with power-of-two 128-block `x_scale`. Weight scales come from
    `hpc_prepare_fp8_moe_weights_sm120`.
    """
    from hpc import fuse_moe_blockwise_fp8_sm120

    return fuse_moe_blockwise_fp8_sm120(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
        output=output,
        swiglu_limit=swiglu_limit,
    )


def hpc_fuse_moe_blockwise_mxfp4_sm120(
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
    """MXFP4 weight x MXFP8 activation fused MoE on SM120.

    `x` is MXFP8 with its UE8M0 `x_scale`, or BF16 with `x_scale=None`, which
    the kernel quantizes to MXFP8 itself. Weight scales come from
    `hpc_prepare_mxfp4_moe_weights_sm120`.
    """
    from hpc import fuse_moe_blockwise_mxfp4_sm120

    return fuse_moe_blockwise_mxfp4_sm120(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
        output=output,
        swiglu_limit=swiglu_limit,
    )


__all__ = [
    "has_hpc",
    "has_hpc_sm120_moe",
    "hpc_fuse_moe",
    "hpc_fuse_moe_blockwise",
    "hpc_fuse_moe_blockwise_fp8_sm120",
    "hpc_fuse_moe_blockwise_mxfp4_sm120",
    "hpc_prepare_fp8_moe_weights_sm120",
    "hpc_prepare_mxfp4_moe_weights_sm120",
]
