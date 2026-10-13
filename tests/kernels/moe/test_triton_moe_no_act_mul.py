# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE with non-gated activations (*_no_mul).

These tests verify that MoE layers work correctly with activations like
silu_no_mul, gelu_no_mul, relu2_no_mul where the activation output dimension
equals N (not N // 2 like gated activations).
"""

import pytest
import torch

from tests.kernels.moe.utils import (
    check_accuracy,
    make_dummy_moe_config,
    make_test_quant_config,
)
from tests.kernels.quant_utils import dequant
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
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
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


@pytest.mark.parametrize("m", M_SIZES)
@pytest.mark.parametrize("n", N_SIZES)
@pytest.mark.parametrize("k", K_SIZES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("activation", NO_MUL_ACTIVATIONS)
@pytest.mark.parametrize("use_expert_map", [False, True])
@pytest.mark.parametrize("quant_dtype", [None, current_platform.fp8_dtype()])
@torch.inference_mode()
def test_triton_experts_no_mul_activation(
    m: int,
    n: int,
    k: int,
    topk: int,
    activation: MoEActivation,
    use_expert_map: bool,
    quant_dtype: torch.dtype | None,
):
    if quant_dtype is not None and not current_platform.supports_fp8():
        pytest.skip("FP8 is not supported on this device")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    hidden_states = torch.randn(m, k, dtype=dtype, device=DEVICE)
    w1, w2, quant_config = make_test_quant_config(
        NUM_EXPERTS, n, k, dtype, quant_dtype, make_gate=False
    )
    topk_weights = torch.full((m, topk), 1 / topk, device=DEVICE)
    topk_ids = torch.randint(0, NUM_EXPERTS, (m, topk), device=DEVICE)
    topk_ids[::2, 0] = -1
    expert_map = (
        torch.arange(NUM_EXPERTS, dtype=torch.int32, device=hidden_states.device)
        if use_expert_map
        else None
    )

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=quant_config,
    )
    qhidden_states, a1_scale = moe_kernel_quantize_input(
        hidden_states, None, quant_dtype, False
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

    workspace1 = torch.empty(ws1_shape, dtype=dtype, device=DEVICE)
    workspace2 = torch.full(ws2_shape, 1e4, dtype=dtype, device=DEVICE)
    output = torch.empty(out_shape, dtype=dtype, device=DEVICE)

    experts.apply(
        output=output,
        hidden_states=qhidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=NUM_EXPERTS,
        expert_map=expert_map,
        a1q_scale=a1_scale,
        a2_scale=None,
        workspace13=workspace1,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    ids = topk_ids.flatten().clamp_min(0)
    a = dequant(qhidden_states, a1_scale, None, False).repeat_interleave(topk, 0)
    w1_ref = dequant(w1, quant_config.w1_scale, None, False)
    projected = torch.bmm(w1_ref[ids], a.unsqueeze(-1)).squeeze(-1).to(dtype)
    projected[topk_ids.flatten() < 0] = 0
    activated = torch.empty_like(projected)
    apply_moe_activation(activation, activated, projected)
    aq, scale = moe_kernel_quantize_input(activated, None, quant_dtype, False)
    w2_ref = dequant(w2, quant_config.w2_scale, None, False)
    expected = torch.bmm(
        w2_ref[ids], dequant(aq, scale, None, False).unsqueeze(-1)
    ).view(m, topk, k)
    expected = (expected * topk_weights.unsqueeze(-1)).to(dtype).sum(dim=1)
    if quant_dtype is None:
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    else:
        # The kernel accumulates the first GEMM in another order, so an activation
        # on an fp8 rounding boundary can quantize to its neighbour and move the
        # outputs it feeds by a whole fp8 step. Unwritten slots or a polluted
        # activation scale corrupt most outputs instead.
        check_accuracy(output, expected, atol=2e-2, rtol=2e-2, percent=0.9)


@torch.inference_mode()
def test_workspace_shapes_no_mul_vs_gated():
    """Test that workspace shapes differ correctly between gated and non-gated."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    M, N, K, topk = 64, 256, 128, 2

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    ws1_no_mul, _, out_no_mul = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, MoEActivation.SILU_NO_MUL
    )

    ws1_gated, _, out_gated = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, MoEActivation.SILU
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


@torch.inference_mode()
def test_adjust_n_for_activation():
    """Test the adjust_N_for_activation method."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    N = 256

    # Gated activations should return N // 2
    assert experts.adjust_N_for_activation(N, MoEActivation.SILU) == N // 2
    assert experts.adjust_N_for_activation(N, MoEActivation.GELU) == N // 2

    # Non-gated activations should return N
    assert experts.adjust_N_for_activation(N, MoEActivation.SILU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, MoEActivation.GELU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, MoEActivation.RELU2_NO_MUL) == N
