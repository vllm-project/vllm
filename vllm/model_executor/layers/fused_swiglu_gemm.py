# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused dense gate/up projection and SwiGLU kernels."""

from __future__ import annotations

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_SUPPORTED_TOKEN_COUNTS = frozenset((1024, 4096, 8192))
_SUPPORTED_INPUT_SIZE = 1024
_SUPPORTED_INTERMEDIATE_SIZE = 3584


@triton.jit
def _fused_swiglu_gemm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    M,
    N,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        gate_weight = tl.load(
            weight_ptr + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        up_weight = tl.load(
            weight_ptr + (offs_n[None, :] + N) * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        accumulator_gate += tl.dot(x, gate_weight)
        accumulator_up += tl.dot(x, up_weight)

    output = accumulator_gate * tl.sigmoid(accumulator_gate) * accumulator_up
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        output,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_swiglu_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0] // 2
    use_kernel = (
        m in _SUPPORTED_TOKEN_COUNTS
        and k == _SUPPORTED_INPUT_SIZE
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and weight.shape == (2 * _SUPPORTED_INTERMEDIATE_SIZE, _SUPPORTED_INPUT_SIZE)
        and weight.dtype == torch.bfloat16
        and weight.device == x.device
        and weight.is_contiguous()
    )
    if not use_kernel:
        gate, up = torch.nn.functional.linear(x, weight).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 32))
    _fused_swiglu_gemm_kernel[grid](
        x,
        weight,
        output,
        M=m,
        N=n,
        K=k,
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    return output


def fused_swiglu_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="fused_swiglu_gemm",
    op_func=fused_swiglu_gemm_impl,
    fake_impl=fused_swiglu_gemm_fake,
)


def fused_swiglu_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.fused_swiglu_gemm(x, weight)


def supports_fused_swiglu_gemm(layer: MergedColumnParallelLinear) -> bool:
    """Return whether a layer satisfies the static SM120 BF16 contract."""
    config = get_current_vllm_config_or_none()
    linear_backend = (
        config.kernel_config.linear_backend if config is not None else "auto"
    )
    weight = getattr(layer, "weight", None)
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and linear_backend == "auto"
        and isinstance(layer.quant_method, UnquantizedLinearMethod)
        and layer.tp_size == 1
        and layer.bias is None
        and isinstance(weight, torch.Tensor)
        and weight.shape == (2 * _SUPPORTED_INTERMEDIATE_SIZE, _SUPPORTED_INPUT_SIZE)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )
