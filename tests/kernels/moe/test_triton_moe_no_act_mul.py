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
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

pytestmark = [
    pytest.mark.skipif(
        not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
        reason="Triton MoE kernels require a CUDA-alike or XPU device.",
    ),
    pytest.mark.skipif(
        current_platform.is_cuda_alike()
        and not current_platform.has_device_capability(80),
        reason="Triton MoE kernels require compute capability >= 8.0 on CUDA/ROCm.",
    ),
]

# Test parameters
M_SIZES = [1, 16, 64]
N_SIZES = [128, 256]
K_SIZES = [64, 128]
TOPK_VALUES = [1, 2]
NUM_EXPERTS = 8
NO_MUL_ACTIVATIONS = [
    MoEActivation.SILU_NO_MUL,
    MoEActivation.GELU_NO_MUL,
    MoEActivation.RELU2_NO_MUL,
]


def test_apply_moe_activation_supported_contract():
    supported = {
        activation
        for activation in MoEActivation
        if apply_moe_activation_supported(activation)
    }

    assert supported == set(MoEActivation) - {MoEActivation.RELU2}


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "activation",
    [
        activation
        for activation in MoEActivation
        if apply_moe_activation_supported(activation)
    ],
)
@torch.inference_mode()
def test_supported_apply_moe_activation_executes(activation: MoEActivation):
    output_width = 64
    input_width = output_width * 2 if activation.is_gated else output_width
    input = torch.randn(2, input_width, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(2, output_width, device="cuda", dtype=torch.bfloat16)
    activation_config = ApplyMoEActivationConfig()
    if activation == MoEActivation.SITU:
        activation_config = ApplyMoEActivationConfig(activation_situ_beta=1.0)
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        activation_config = ApplyMoEActivationConfig(clamp_limit=7.0)

    apply_moe_activation(activation, output, input, activation_config=activation_config)

    assert torch.isfinite(output).all()


@pytest.mark.parametrize("activation", list(MoEActivation))
def test_triton_activation_metadata_tracks_shared_apply(activation: MoEActivation):
    assert TritonExperts._supports_activation(
        activation
    ) == apply_moe_activation_supported(activation)


def make_test_tensors(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEVICE,
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


@pytest.mark.parametrize("m", M_SIZES)
@pytest.mark.parametrize("n", N_SIZES)
@pytest.mark.parametrize("k", K_SIZES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("activation", NO_MUL_ACTIVATIONS)
@pytest.mark.parametrize("use_expert_map", [False, True])
@torch.inference_mode()
def test_triton_experts_no_mul_activation(
    m: int,
    n: int,
    k: int,
    topk: int,
    activation: MoEActivation,
    use_expert_map: bool,
):
    hidden_states, w1, w2, topk_weights, topk_ids = make_test_tensors(
        m, n, k, NUM_EXPERTS, topk
    )
    topk_ids[::2, 0] = -1
    expert_map = (
        torch.arange(NUM_EXPERTS, dtype=torch.int32, device=hidden_states.device)
        if use_expert_map
        else None
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

    workspace1 = torch.empty(
        ws1_shape[0] * ws1_shape[1] * ws1_shape[2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    workspace2 = torch.full(
        (ws2_shape[0] * ws2_shape[1] * ws2_shape[2],),
        float("nan"),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = torch.zeros(
        out_shape, dtype=hidden_states.dtype, device=hidden_states.device
    )

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=NUM_EXPERTS,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace1,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    expected = torch.zeros_like(output)
    for expert in range(NUM_EXPERTS):
        projected = hidden_states @ w1[expert].T
        activated = torch.empty_like(projected)
        apply_moe_activation(activation, activated, projected)
        weights = (topk_weights * (topk_ids == expert)).sum(dim=1, keepdim=True)
        expected += (activated @ w2[expert].T) * weights.to(output.dtype)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_workspace_shapes_no_mul_vs_gated():
    M, N, K, topk = 64, 256, 128, 2

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    ws1_no_mul, ws2_no_mul, out_no_mul = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, MoEActivation.SILU_NO_MUL
    )

    ws1_gated, ws2_gated, out_gated = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, MoEActivation.SILU
    )

    assert ws1_no_mul == (M, topk, max(N, K))
    assert ws1_gated == (M, topk, max(N // 2, K))
    assert ws2_no_mul == ws2_gated == (M, topk, max(N, K))
    assert out_no_mul == out_gated == (M, K)


@torch.inference_mode()
def test_adjust_n_for_activation():
    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    N = 256

    assert experts.adjust_N_for_activation(N, MoEActivation.SILU) == N // 2
    assert experts.adjust_N_for_activation(N, MoEActivation.GELU) == N // 2

    assert experts.adjust_N_for_activation(N, MoEActivation.SILU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, MoEActivation.GELU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, MoEActivation.RELU2_NO_MUL) == N
