# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def is_rocm_aiter_rmsnorm_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER_RMSNORM and envs.VLLM_ROCM_USE_AITER


def rocm_aiter_rmsnorm_fused_dynamic_quant_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    yscale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_dynamicquant(
        out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm=0
    )

    return out, yscale


def rocm_aiter_rmsnorm_fused_dynamic_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    yscale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return out, yscale


def rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    yscale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_add_dynamicquant(
        out,
        input,
        residual,
        residual_out,
        yscale,
        weight,
        epsilon,
        use_model_sensitive_rmsnorm=0,
    )

    return out, yscale, residual_out


def rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    yscale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return out, yscale, residual_out


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm_fused_dynamic_quant",
        op_func=rocm_aiter_rmsnorm_fused_dynamic_quant_impl,
        mutates_args=["out", "yscale"],
        fake_impl=rocm_aiter_rmsnorm_fused_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm_fused_add_dynamic_quant",
        op_func=rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl,
        mutates_args=["out", "yscale", "residual_out"],
        fake_impl=rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
