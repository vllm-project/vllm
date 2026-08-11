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
    make_lut_b_moe_kernel,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    dequantize_lut_b,
    quantize_lut_b,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="The LUT-B fused MoE path requires a CUDA-like GPU.",
)
def test_lut_b_fused_moe_matches_dequantized_oracle(workspace_init) -> None:
    """Decode-in-GEMM matches the same routed MoE over reconstructed weights."""
    torch.manual_seed(1)
    num_tokens = 7
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
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0]],
        device="cuda",
        dtype=torch.int64,
    )
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
