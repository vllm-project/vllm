# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for fused LUT-B MoE expert GEMMs."""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    dequantize_lut_b_triton,
    make_lut_b_moe_kernel,
    should_dequantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    dequantize_lut_b,
    quantize_lut_b,
)
from vllm.platforms import current_platform


def test_lut_b_dequant_heuristic_boundary() -> None:
    assert not should_dequantize_lut_b(num_tokens=7, num_experts=4)
    assert should_dequantize_lut_b(num_tokens=8, num_experts=4)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="The LUT-B dequantization kernel requires a CUDA-like GPU.",
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_lut_b_triton_dequant_matches_oracle(out_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    weight = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    packed, codebooks = quantize_lut_b(weight)

    actual = dequantize_lut_b_triton(packed, codebooks, out_dtype)
    expected = dequantize_lut_b(packed, codebooks, out_dtype=out_dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="The LUT-B fused MoE path requires a CUDA-like GPU.",
)
@pytest.mark.parametrize("num_tokens", [7, 8])
def test_lut_b_fused_moe_matches_dequantized_oracle(
    workspace_init,
    num_tokens: int,
) -> None:
    """Both execution paths match routed MoE over reconstructed weights."""
    torch.manual_seed(1)
    num_experts = 4
    top_k = 2
    hidden_size = 64
    intermediate_size = 64

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 10
    )
    token_ids = torch.arange(num_tokens, device="cuda")
    topk_ids = torch.stack((token_ids % num_experts, (token_ids + 1) % num_experts), 1)
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        device="cuda",
        dtype=torch.float32,
    )

    w13_packed, w13_codebooks = quantize_lut_b(w13)
    w2_packed, w2_codebooks = quantize_lut_b(w2)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc("lut_b", scale=w13_codebooks),
        _w2=FusedMoEQuantDesc("lut_b", scale=w2_codebooks),
    )
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        in_dtype=hidden_states.dtype,
        activation=MoEActivation.SILU,
    )
    kernel = make_lut_b_moe_kernel(moe_config, quant_config, None)
    actual = kernel.apply(
        hidden_states,
        w13_packed,
        w2_packed,
        topk_weights,
        topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    w13_dequant = dequantize_lut_b(
        w13_packed,
        w13_codebooks,
        out_dtype=hidden_states.dtype,
    )
    w2_dequant = dequantize_lut_b(
        w2_packed,
        w2_codebooks,
        out_dtype=hidden_states.dtype,
    )
    expected = torch.zeros_like(hidden_states)
    for token in range(num_tokens):
        for route in range(top_k):
            expert = topk_ids[token, route]
            hidden = F.linear(hidden_states[token], w13_dequant[expert])
            gate, up = hidden.chunk(2)
            hidden = F.silu(gate) * up
            hidden = F.linear(hidden, w2_dequant[expert])
            expected[token] += hidden * topk_weights[token, route]

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
