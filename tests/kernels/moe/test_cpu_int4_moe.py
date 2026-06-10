# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU INT4 W4A8 dynamic quantized fused MoE kernel (CPUExpertsInt4)."""

import sys

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Check if the dynamic_4bit_int_moe op is available
if not hasattr(torch.ops._C, "dynamic_4bit_int_moe"):
    pytest.skip("dynamic_4bit_int_moe op not available", allow_module_level=True)

# Check if KleidiAI ops are available
if not hasattr(torch.ops.aten, "_dyn_quant_pack_4bit_weight"):
    pytest.skip("KleidiAI 4-bit ops not available", allow_module_level=True)


# Tolerance for INT4 W4A8
INT4_W4A8_ATOL = 2e-2
INT4_W4A8_RTOL = 2e-2


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: SiLU(gate) * up."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _pack_int4_weight_to_kleidi(
    int4_as_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Pack INT4 weights (stored as int8 in [-8,7]) to KleidiAI format.

    Args:
        int4_as_int8: [out, in] int8 tensor with values in [-8, 7]
        scales: [out, in//group_size] or [out, 1] for channel-wise
        bias: [out] optional bias
        group_size: Quantization group size (-1 for channel-wise)
        in_features: Input dimension
        out_features: Output dimension

    Returns:
        Packed weight tensor in KleidiAI format
    """
    # Shift to unsigned nibble [0, 15]
    tmp = int4_as_int8.add(8)
    # Pack pairs along input dimension
    uint8_nibbles = ((tmp[:, 1::2] << 4) | tmp[:, ::2]).to(torch.uint8)

    # Determine scale dtype based on group_size
    scale_dtype = torch.float32 if group_size == -1 else torch.bfloat16
    scales_typed = scales.to(scale_dtype)
    bias_typed = None if bias is None else bias.to(torch.float32)

    # Pack using KleidiAI op
    actual_group_size = in_features if group_size == -1 else group_size
    return torch.ops.aten._dyn_quant_pack_4bit_weight(
        uint8_nibbles,
        scales_typed,
        bias_typed,
        actual_group_size,
        in_features,
        out_features,
    )


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

    # Pack weights for each expert
    w13_packed_list = []
    w2_packed_list = []

    for e in range(E):
        w13_packed_list.append(
            _pack_int4_weight_to_kleidi(
                w13_int4[e],
                w13_scales[e],
                w13_bias[e] if (has_bias and w13_bias is not None) else None,
                group_size,
                K,
                2 * N,
            )
        )
        w2_packed_list.append(
            _pack_int4_weight_to_kleidi(
                w2_int4[e],
                w2_scales[e],
                w2_bias[e] if (has_bias and w2_bias is not None) else None,
                group_size,
                N,
                K,
            )
        )

    w13_packed = torch.stack(w13_packed_list, dim=0)
    w2_packed = torch.stack(w2_packed_list, dim=0)

    # Create reference dequantized weights
    w13_ref = torch.zeros(E, 2 * N, K, dtype=torch.float32)
    w2_ref = torch.zeros(E, K, N, dtype=torch.float32)

    for e in range(E):
        # Dequantize w13
        for i in range(2 * N):
            for j in range(K):
                group_idx = 0 if group_size == -1 else (j // group_size)
                w13_ref[e, i, j] = (
                    w13_int4[e, i, j].float() * w13_scales[e, i, group_idx].float()
                )
                if has_bias and w13_bias is not None:
                    w13_ref[e, i, j] += w13_bias[e, i].float()

        # Dequantize w2
        for i in range(K):
            for j in range(N):
                group_idx = 0 if group_size == -1 else (j // group_size)
                w2_ref[e, i, j] = (
                    w2_int4[e, i, j].float() * w2_scales[e, i, group_idx].float()
                )
                if has_bias and w2_bias is not None:
                    w2_ref[e, i, j] += w2_bias[e, i].float()

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


@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("N,K,E,topk,group_size", MoE_CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cpu_int4_moe_kernel(M, N, K, E, topk, group_size, seed):
    """Test dynamic_4bit_int_moe kernel against dequantized torch reference."""
    set_random_seed(seed)

    # Generate input activations
    a = torch.randn(M, K, dtype=torch.bfloat16) / (K**0.5)

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
    # Activation kind: 1 = SwiGLU_Ug (SiLU(u)*g) for OAI-style
    activation_kind = 1
    apply_router_weight_on_input = False

    out = torch.ops._C.dynamic_4bit_int_moe(
        a,
        topk_ids,
        topk_weight,
        w13_packed,
        w2_packed,
        K,  # H (hidden_size / w2_out_features)
        N,  # I (intermediate_size / w2_in_features)
        2 * N,  # I2 (2*intermediate_size / w13_out_features)
        group_size,
        apply_router_weight_on_input,
        activation_kind,
    )

    torch.testing.assert_close(
        ref_out.bfloat16(),
        out,
        atol=INT4_W4A8_ATOL,
        rtol=INT4_W4A8_RTOL,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
