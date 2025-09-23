# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""
from typing import Optional, Union

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def is_rocm_aiter_rmsnorm_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER_RMSNORM \
        and envs.VLLM_ROCM_USE_AITER

def rocm_aiter_rmsnorm_fused_dynamic_quant_impl(output: torch.Tensor,
                                                input: torch.Tensor, 
                                                weight: torch.Tensor,
                                                scale: torch.Tensor,
                                                epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_dynamicquant(
        output, input, scale, weight, epsilon, use_model_sensitive_rmsnorm=0
    )

    return output, scale

def rocm_aiter_rmsnorm_fused_dynamic_quant_fake(output: torch.Tensor,
                                                input: torch.Tensor, 
                                                weight: torch.Tensor,
                                                scale: torch.Tensor,
                                                epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    return output, scale

def rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl(output: torch.Tensor,
                                                    input: torch.Tensor,
                                                    residual: torch.Tensor,
                                                    residual_out: torch.Tensor,
                                                    weight: torch.Tensor,
                                                    scale: torch.Tensor,
                                                    epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_add_dynamicquant(
            output,
            input,
            residual,
            residual_out,
            scale,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm=0,
        )


    return output, scale, residual_out

def rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake(output: torch.Tensor,
                                                    input: torch.Tensor,
                                                    residual: torch.Tensor,
                                                    residual_out: torch.Tensor,
                                                    weight: torch.Tensor,
                                                    scale: torch.Tensor,
                                                    epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    return output, scale, residual_out

if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm_fused_dynamic_quant",
        op_func=rocm_aiter_rmsnorm_fused_dynamic_quant_impl,
        mutates_args=['output', 'scale'],
        fake_impl=rocm_aiter_rmsnorm_fused_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm_fused_add_dynamic_quant",
        op_func=rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl,
        mutates_args=['output', 'scale', 'residual_out'],
        fake_impl=rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )