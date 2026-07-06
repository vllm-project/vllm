# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU INT4 W4A8 dynamic quantized fused MoE kernel (CPUExpertsInt4)."""

import sys

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import (
    CPUExpertsInt4,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import (
    convert_to_w4a8_int8_moe_format,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed

if (
    not current_platform.is_cpu()
    or current_platform.get_cpu_architecture() != CpuArchEnum.ARM
):
    pytest.skip("skipping Arm CPU-only tests", allow_module_level=True)


# Tolerance for INT4 W4A8
INT4_W4A8_ATOL = 2e-2
INT4_W4A8_RTOL = 2e-2


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: SiLU(gate) * up."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _make_int4_moe_weights(
    E: int,
    N: int,
    K: int,
    group_size: int,
    has_bias: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Generate random INT4 MoE weights with random scales.

    Args:
        E: Number of experts
        N: Intermediate size
        K: Hidden size
        group_size: Quantization group size (-1 for channel-wise)
        has_bias: Whether to include bias

    Returns:
        (w13_packed, w2_packed, w13_ref, w2_ref, w13_bias, w2_bias)
        where *_ref are the dequantized float reference weights
    """
    # Generate INT4 weights as int8 values in [-8, 7]
    w13_int4 = torch.randint(-8, 8, (E, 2 * N, K), dtype=torch.int8)
    w2_int4 = torch.randint(-8, 8, (E, K, N), dtype=torch.int8)

    # Determine number of scale columns
    def _n_scale_cols(in_features: int) -> int:
        return 1 if group_size == -1 else (in_features // group_size)

    # Generate random scales
    scale_dtype = torch.float32 if group_size == -1 else torch.bfloat16
    w13_scales = torch.rand(E, 2 * N, _n_scale_cols(K), dtype=scale_dtype) * 0.01
    w2_scales = torch.rand(E, K, _n_scale_cols(N), dtype=scale_dtype) * 0.01

    # Generate biases if needed
    w13_bias = None
    w2_bias = None
    if has_bias:
        w13_bias = torch.randn(E, 2 * N, dtype=torch.float32) * 0.01
        w2_bias = torch.randn(E, K, dtype=torch.float32) * 0.01

    w13_packed, w2_packed, *_ = convert_to_w4a8_int8_moe_format(
        w13_weight=w13_int4,
        w2_weight=w2_int4,
        w13_weight_scale=w13_scales,
        w2_weight_scale=w2_scales,
        group_size=group_size,
        w13_bias=w13_bias if has_bias else None,
        w2_bias=w2_bias if has_bias else None,
    )

    if group_size == -1:
        w13_scale = w13_scales.float()
        w2_scale = w2_scales.float()
    else:
        w13_scale = w13_scales.float().repeat_interleave(group_size, dim=-1)
        w2_scale = w2_scales.float().repeat_interleave(group_size, dim=-1)

    w13_ref = w13_int4.float() * w13_scale
    w2_ref = w2_int4.float() * w2_scale
    if has_bias and w13_bias is not None:
        w13_ref = w13_ref + w13_bias.float().unsqueeze(-1)
    if has_bias and w2_bias is not None:
        w2_ref = w2_ref + w2_bias.float().unsqueeze(-1)

    return w13_packed, w2_packed, w13_ref, w2_ref, w13_bias, w2_bias


def ref_int4_moe(
    a: torch.Tensor,
    w13_ref: torch.Tensor,
    w2_ref: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference INT4 W4A8 fused MoE using dequantized weights.

    Steps:
      1. Use dequantized float weights
      2. For each expert: matmul → SwiGLU → matmul
      3. Weighted sum across top-k experts
    """
    B, D = a.shape
    topk = topk_ids.size(1)

    a_exp = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
    out = torch.zeros(B * topk, w2_ref.shape[1], dtype=torch.float32)

    topk_weight_flat = topk_weight.view(-1)
    topk_ids_flat = topk_ids.view(-1)

    for i in range(w13_ref.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum():
            # w13: [2N, K], input: [B, K] -> output: [B, 2N]
            gate_up = torch.matmul(a_exp[mask], w13_ref[i].transpose(0, 1))
            # SwiGLU activation
            hidden = _silu_and_mul(gate_up)
            # w2: [K, N], hidden: [B, N] -> output: [B, K]
            out[mask] = torch.matmul(hidden, w2_ref[i].transpose(0, 1))

    return (
        (out.view(B, -1, w2_ref.shape[1]) * topk_weight_flat.view(B, -1, 1))
        .sum(dim=1)
        .to(a.dtype)
    )


NUM_TOKENS = [1, 2, 64, 128]
# (intermediate_size N, hidden_size K, num_experts E, topk, group_size)
MoE_CONFIGS = [
    (256, 512, 8, 2, 128),
    (256, 512, 8, 2, 64),
    (256, 512, 8, 2, -1),  # channel-wise
    (512, 256, 8, 4, 128),
    (512, 512, 8, 2, 128),
    (768, 2048, 8, 2, 128),
    (768, 2048, 16, 4, 64),
]
SEEDS = [0, 42]
ACTIVATION_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk,group_size", MoE_CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("activation_dtype", ACTIVATION_DTYPES)
def test_cpu_int4_moe_kernel(M, N, K, E, topk, group_size, seed, activation_dtype):
    """Test dynamic_4bit_int_moe kernel against dequantized torch reference."""
    set_random_seed(seed)
    activation = MoEActivation.SILU

    # Generate input activations
    a = torch.randn(M, K, dtype=activation_dtype) / (K**0.5)

    # Generate INT4 weights
    w13_packed, w2_packed, w13_ref, w2_ref, w13_bias, w2_bias = _make_int4_moe_weights(
        E, N, K, group_size, has_bias=False
    )

    # Generate router logits and topk
    score = torch.randn(M, E, dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.long)

    # Reference output using dequantized weights
    ref_out = ref_int4_moe(
        a,
        w13_ref,
        w2_ref,
        topk_weight,
        topk_ids,
    )

    # Test dynamic_4bit_int_moe kernel
    apply_router_weight_on_input = False

    out = torch.ops._C.dynamic_4bit_int_moe(
        a,
        topk_ids,
        topk_weight,
        w13_packed,
        w2_packed,
        K,  # H (hidden_size / w2_out_features)
        N,  # I (intermediate_size / w2_in_features)
        group_size,
        apply_router_weight_on_input,
        CPUExpertsInt4._activation_kind(activation),
    )

    assert out.dtype == activation_dtype
    torch.testing.assert_close(
        ref_out,
        out,
        atol=INT4_W4A8_ATOL,
        rtol=INT4_W4A8_RTOL,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
