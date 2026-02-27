# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ZeroExpertFusedMoE.

Verifies that:
- The ZeroExpertRouter is properly created and used as the layer router.
- A forward pass through ZeroExpertFusedMoE produces correct output.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context, set_forward_context
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
    hidden_size = 128
    intermediate_size = 256
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
