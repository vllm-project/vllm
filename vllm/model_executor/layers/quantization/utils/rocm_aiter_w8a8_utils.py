# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx
from vllm.utils import direct_register_custom_op


def is_rocm_aiter_gemm_enabled():
    return current_platform.is_rocm() \
            and on_mi3xx() \
            and envs.VLLM_ROCM_USE_AITER \
            and envs.VLLM_ROCM_USE_AITER_LINEAR


def rocm_aiter_gemm_a8w8_bpreshuffle_impl(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:

    # This AITER function can be used for
    # - per-token activations + per-channel weights
    #   e.g. vllm/model_executor/layers/quantization/utils/w8a8_utils.py
    # accept the weight as # keep the weight as (N, K)
    # NOTE: The weight has to be shuffled in the
    # process_weights_after_loading of the CompressedTensorsW8A8Fp8 class

    from aiter import gemm_a8w8_bpreshuffle_CK

    return gemm_a8w8_bpreshuffle_CK(qinput, weight, scale_a, scale_b,
                                    out_dtype)


def rocm_aiter_gemm_a8w8_bpreshuffle_fake(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:

    m = qinput.shape[0]
    n = weight.shape[0]
    if out_dtype is None:
        out_dtype = qinput.dtype
    return torch.empty((m, n), dtype=out_dtype, device=qinput.device)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_gemm_a8w8_bpreshuffle",
        op_func=rocm_aiter_gemm_a8w8_bpreshuffle_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_gemm_a8w8_bpreshuffle_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def rocm_aiter_per_token_w8a8_scaled_mm(qinput: torch.Tensor,
                                        weight: torch.Tensor,
                                        out_dtype: torch.dtype,
                                        scale_a: torch.Tensor,
                                        scale_b: torch.Tensor,
                                        bias: torch.Tensor,
                                        input_2d: torch.Tensor,
                                        output_shape: list) -> torch.Tensor:
    output_shape = [*qinput.shape[:-1], weight.shape[0]]
    output = torch.ops.vllm.rocm_aiter_gemm_a8w8_bpreshuffle(
        qinput, weight, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
    if bias is not None:
        output = output + bias

    return torch.narrow(output, 0, 0, input_2d.shape[0]).view(*output_shape)
