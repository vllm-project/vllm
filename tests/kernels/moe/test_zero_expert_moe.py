# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ZeroExpertFusedMoE.

Verifies that:
- The ZeroExpertRouter is properly created and used as the layer router.
- A forward pass through ZeroExpertFusedMoE produces correct output.
- The output decomposes correctly into real expert + zero expert contributions.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE,
)
from vllm.v1.worker.workspace import init_workspace_manager


@pytest.fixture
def zero_expert_moe(dist_init, default_vllm_config):
    """Create a ZeroExpertFusedMoE layer with zero experts."""
    num_experts = 4
    top_k = 2
    # hidden_size must be >= 256 for the zero expert identity kernel to
    # produce output (its BLOCK_SIZE=256 causes grid=0 when hidden_dim<256).
    hidden_size = 256
    intermediate_size = 512
    zero_expert_num = 1

    e_score_correction_bias = torch.zeros(
        num_experts + zero_expert_num,
        dtype=torch.float32,
        device="cuda",
    )

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.cuda.current_device())

        layer = ZeroExpertFusedMoE(
            zero_expert_num=zero_expert_num,
            zero_expert_type="identity",
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            prefix="test_zero_expert_moe",
            renormalize=False,
            routed_scaling_factor=1.0,
            scoring_func="softmax",
        ).cuda()

        layer.quant_method.process_weights_after_loading(layer)

        yield layer, vllm_config


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_zero_expert_moe_router_is_zero_expert_router(zero_expert_moe, num_tokens):
    """Verify that ZeroExpertFusedMoE creates a ZeroExpertRouter."""
    layer, _ = zero_expert_moe
    assert isinstance(layer.router, ZeroExpertRouter), (
        f"Expected ZeroExpertRouter but got {type(layer.router).__name__}."
    )


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_zero_expert_moe_no_custom_routing_fn(zero_expert_moe, num_tokens):
    """Verify that custom_routing_function is not set (routing is handled
    by ZeroExpertRouter, not a memoizing closure)."""
    layer, _ = zero_expert_moe
    assert layer.custom_routing_function is None


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_zero_expert_moe_forward(zero_expert_moe, num_tokens):
    """Run a forward pass through ZeroExpertFusedMoE and verify output shape."""
    layer, vllm_config = zero_expert_moe

    hidden_size = layer.hidden_size
    num_experts = 4
    zero_expert_num = 1
    total_experts = num_experts + zero_expert_num

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = torch.randn(
        num_tokens, total_experts, dtype=torch.float32, device="cuda"
    )

    # Initialize weights to small random values to avoid NaN from
    # uninitialized memory.
    with torch.no_grad():
        for param in layer.parameters():
            if param.dtype.is_floating_point:
                param.normal_(0, 0.01)

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        get_forward_context().all_moe_layers = None
        output = layer.forward(hidden_states, router_logits)

    assert output.shape == hidden_states.shape, (
        f"Expected output shape {hidden_states.shape}, got {output.shape}"
    )
    assert output.dtype == hidden_states.dtype
    assert not torch.isnan(output).any(), "Output contains NaN values"


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_zero_expert_moe_output_decomposition(zero_expert_moe, num_tokens):
    """Validate that ZeroExpertFusedMoE output equals real expert output
    plus zero expert contribution.

    The key invariant is:
        layer.forward(h, r) == FusedMoE.forward(h, r) + zero_expert_output

    FusedMoE.forward() computes only the real expert MoE output (the
    ZeroExpertRouter masks zero expert entries to weight=0), while the
    zero expert contribution is computed as a side effect during routing
    and added on top by ZeroExpertFusedMoE.forward().
    """
    layer, vllm_config = zero_expert_moe
    num_experts = 4
    zero_expert_num = 1
    total_experts = num_experts + zero_expert_num

    hidden_states = torch.randn(
        num_tokens, layer.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = torch.randn(
        num_tokens, total_experts, dtype=torch.float32, device="cuda"
    )

    with torch.no_grad():
        for param in layer.parameters():
            if param.dtype.is_floating_point:
                param.normal_(0, 0.01)

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        get_forward_context().all_moe_layers = None

        # Get the real expert output only (bypasses ZeroExpertFusedMoE.forward,
        # calls FusedMoE.forward directly). The ZeroExpertRouter still runs and
        # stores zero_expert_output as a side effect.
        real_output = FusedMoE.forward(layer, hidden_states, router_logits)
        zero_output = layer.router.zero_expert_output

        # Get the full combined output.
        full_output = layer.forward(hidden_states, router_logits)

    assert zero_output is not None, "Zero expert output should not be None"
    assert not torch.isnan(real_output).any(), "Real expert output has NaN"
    assert not torch.isnan(zero_output).any(), "Zero expert output has NaN"
    assert not torch.isnan(full_output).any(), "Full output has NaN"

    expected = real_output + zero_output
    torch.testing.assert_close(
        full_output,
        expected,
        atol=0,
        rtol=0,
        msg="ZeroExpertFusedMoE output should equal real expert output "
        "plus zero expert contribution",
    )


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_zero_expert_moe_zero_expert_is_identity(zero_expert_moe, num_tokens):
    """Validate zero expert identity behavior.

    When routing strongly favors the zero expert, its contribution should
    be a scaled version of hidden_states (identity operation). We verify
    this by manually computing the expected zero expert output from the
    routing weights and comparing against what the router produces.
    """
    layer, vllm_config = zero_expert_moe
    num_experts = 4
    zero_expert_num = 1
    total_experts = num_experts + zero_expert_num

    hidden_states = torch.randn(
        num_tokens, layer.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    # Strongly bias toward the zero expert (index 4).
    router_logits = torch.full(
        (num_tokens, total_experts), -10.0, dtype=torch.float32, device="cuda"
    )
    router_logits[:, num_experts] = 10.0  # zero expert gets high logit

    with torch.no_grad():
        for param in layer.parameters():
            if param.dtype.is_floating_point:
                param.normal_(0, 0.01)

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        get_forward_context().all_moe_layers = None

        # Run routing to get topk_weights/topk_ids before masking.
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            fused_topk_bias,
        )

        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            e_score_correction_bias=layer.router.e_score_correction_bias.data,
            topk=layer.top_k,
            renormalize=layer.router.renormalize,
            scoring_func=layer.router.scoring_func,
        )

        # Manually compute expected zero expert identity output:
        # For each token, sum routing weights assigned to zero expert slots,
        # then multiply by hidden_states.
        zero_mask = topk_ids >= num_experts
        zero_weight_per_token = (topk_weights * zero_mask.float()).sum(
            dim=-1, keepdim=True
        )
        expected_zero_output = (hidden_states.float() * zero_weight_per_token).to(
            hidden_states.dtype
        )

        # Run the layer forward to trigger routing and get the actual
        # zero expert output from the router.
        FusedMoE.forward(layer, hidden_states, router_logits)
        actual_zero_output = layer.router.zero_expert_output

    assert actual_zero_output is not None
    assert zero_mask.any(), (
        "With high zero expert logit, at least some slots should route "
        "to the zero expert"
    )

    torch.testing.assert_close(
        actual_zero_output,
        expected_zero_output,
        atol=1e-3,
        rtol=1e-3,
        msg="Zero expert identity output should equal "
        "hidden_states * sum(zero_expert_weights)",
    )
