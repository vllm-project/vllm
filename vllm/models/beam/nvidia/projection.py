# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed attention projections using shared low-latency GEMMs."""

import functools

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


@functools.cache
def _tinygemm():
    from flashinfer.jit import gen_tinygemm2_module

    return gen_tinygemm2_module().build_and_load()


def use_low_latency_gemm(weight: torch.Tensor) -> bool:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import is_available

    return (
        weight.dtype == torch.bfloat16
        and weight.device.type == "cuda"
        and current_platform.is_device_capability_family(100)
        and weight.shape in ((1792, 5120), (3584, 5120))
        and is_available()
    )


def warmup_packed_projection(weight: torch.Tensor) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_c1_pdl_kernel,
        ll_bf16_gemm_kernel,
    )

    shapes = [(weight.shape[1], weight.shape[0])]
    ll_bf16_gemm_c1_pdl_kernel.warmup(
        shapes=shapes, m_values=[1], output_dtype=torch.bfloat16
    )
    ll_bf16_gemm_kernel.warmup(
        shapes=shapes, m_values=[2, 16], output_dtype=torch.bfloat16
    )
    _tinygemm()


def packed_projection(
    x: torch.Tensor, weight: torch.Tensor, low_latency: bool
) -> torch.Tensor:
    rows = x.shape[0]
    if not low_latency or not 1 <= rows <= 16 or not x.is_contiguous():
        return F.linear(x, weight)
    tiny_max = 8 if weight.shape[0] == 1792 else 4
    if 3 <= rows <= tiny_max:
        output = x.new_empty((rows, weight.shape[0]))
        _tinygemm().tinygemm2_nobias_op(x, weight, output, False)
        return output
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    return ll_bf16_gemm(x, weight, output_dtype=torch.bfloat16)
