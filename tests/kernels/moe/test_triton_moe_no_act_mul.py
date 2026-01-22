# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE with non-gated activations (*_no_mul).

These tests verify that MoE layers work correctly with activations like
silu_no_mul, gelu_no_mul, relu2_no_mul where the activation output dimension
equals N (not N // 2 like gated activations).
"""

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.utils import (
    GELU_NO_MUL,
    RELU2_NO_MUL,
    SILU_NO_MUL,
)
from vllm.platforms import current_platform

# Test parameters
M_SIZES = [1, 16, 64]
N_SIZES = [128, 256]
K_SIZES = [64, 128]
TOPK_VALUES = [1, 2]
NUM_EXPERTS = 8
NO_MUL_ACTIVATIONS = [SILU_NO_MUL, GELU_NO_MUL, RELU2_NO_MUL]


def make_test_tensors(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create test tensors for MoE with non-gated activation.

    For non-gated activations (*_no_mul):
    - w1: (E, N, K) - projects from K to N
    - w2: (E, K, N) - projects from N back to K (note: N, not N//2)
    """
    hidden_states = torch.randn(m, k, dtype=dtype, device=device)

    # For non-gated: w1 projects K -> N, w2 projects N -> K
    w1 = torch.randn(num_experts, n, k, dtype=dtype, device=device) * 0.1
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device) * 0.1

    topk_weights = torch.ones(m, topk, dtype=torch.float32, device=device) / topk
    topk_ids = torch.randint(0, num_experts, (m, topk), device=device)

    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@pytest.mark.parametrize("m", M_SIZES)
@pytest.mark.parametrize("n", N_SIZES)
@pytest.mark.parametrize("k", K_SIZES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("activation", NO_MUL_ACTIVATIONS)
@torch.inference_mode()
def test_triton_experts_no_mul_activation(
    m: int,
    n: int,
    k: int,
    topk: int,
    activation: str,
):
    hidden_states, w1, w2, topk_weights, topk_ids = make_test_tensors(
        m, n, k, NUM_EXPERTS, topk
    )

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    ws1_shape, ws2_shape, out_shape = experts.workspace_shapes(
        M=m,
        N=n,
        K=k,
        topk=topk,
        global_num_experts=NUM_EXPERTS,
        local_num_experts=NUM_EXPERTS,
        expert_tokens_meta=None,
        activation=activation,
    )

    # Verify workspace shapes are correct for no_mul activation
    # workspace1 should handle activation_out_dim = N (not N//2)
    assert ws1_shape == (m, topk, max(n, k)), (
        f"workspace1 shape mismatch: expected {(m, topk, max(n, k))}, got {ws1_shape}"
    )
    # workspace2 should handle max(N, K) for intermediate_cache1/cache3
    assert ws2_shape == (m, topk, max(n, k)), (
        f"workspace2 shape mismatch: expected {(m, topk, max(n, k))}, got {ws2_shape}"
    )
    assert out_shape == (m, k), (
        f"output shape mismatch: expected {(m, k)}, got {out_shape}"
    )

    workspace1 = torch.empty(
        ws1_shape[0] * ws1_shape[1] * ws1_shape[2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    workspace2 = torch.empty(
        ws2_shape[0] * ws2_shape[1] * ws2_shape[2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = torch.zeros(m, k, dtype=hidden_states.dtype, device=hidden_states.device)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=NUM_EXPERTS,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace1,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert output.shape == (m, k), f"Expected shape {(m, k)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output.abs().sum() > 0, "Output is all zeros"


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_workspace_shapes_no_mul_vs_gated():
    """Test that workspace shapes differ correctly between gated and non-gated."""
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

    M, N, K, topk = 64, 256, 128, 2

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    ws1_no_mul, _, out_no_mul = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, SILU_NO_MUL
    )

    ws1_gated, _, out_gated = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, "silu"
    )

    # For no_mul: activation_out_dim = N
    # For gated: activation_out_dim = N // 2
    # workspace1 should use max(activation_out_dim, K)
    activation_out_dim_no_mul = N
    activation_out_dim_gated = N // 2

    assert ws1_no_mul[2] == max(activation_out_dim_no_mul, K), (
        f"no_mul workspace1 last dim should be max({activation_out_dim_no_mul}, {K})"
    )
    assert ws1_gated[2] == max(activation_out_dim_gated, K), (
        f"gated workspace1 last dim should be max({activation_out_dim_gated}, {K})"
    )

    # Output shapes should be the same
    assert out_no_mul == out_gated == (M, K)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_adjust_n_for_activation():
    """Test the adjust_N_for_activation method."""
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    N = 256

    # Gated activations should return N // 2
    assert experts.adjust_N_for_activation(N, "silu") == N // 2
    assert experts.adjust_N_for_activation(N, "gelu") == N // 2

    # Non-gated activations should return N
    assert experts.adjust_N_for_activation(N, SILU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, GELU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, RELU2_NO_MUL) == N
