#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for the token-to-expert routing simulator.

This script demonstrates how to use the routing simulator to test
different routing strategies and analyze their performance, including
integration tests with FusedMoE layer.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routing_simulator import (
    RoutingSimulator)


@pytest.fixture
def device():
    """Fixture to provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("num_tokens", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [64, 1024])
@pytest.mark.parametrize("num_experts", [16, 128])
@pytest.mark.parametrize("top_k", [1, 4])
def test_basic_functionality(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device,
):
    """Test basic functionality of the routing simulator."""
    # Test each routing strategy
    strategies = RoutingSimulator.get_available_strategies()

    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    for strategy in strategies:
        # Simulate routing
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=strategy,
            top_k=top_k,
        )

        # Check output shapes
        assert topk_weights.shape == (
            num_tokens,
            top_k,
        ), f"Wrong weights shape for {strategy}"
        assert topk_ids.shape == (
            num_tokens,
            top_k,
        ), f"Wrong ids shape for {strategy}"

        # Check that expert IDs are valid
        assert (topk_ids.min()
                >= 0), f"Invalid expert ID (negative) for {strategy}"
        assert (topk_ids.max()
                < num_experts), f"Invalid expert ID (too large) for {strategy}"


def test_routing_strategy_integration(monkeypatch, device):
    """Test that the routing strategy environment variable works with
    FusedMoE."""
    pytest.importorskip("vllm.model_executor.layers.fused_moe.layer")

    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Test parameters
    num_tokens = 32
    hidden_size = 16
    num_experts = 4
    top_k = 2

    # Create test data
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Test different routing strategies
    strategies = RoutingSimulator.get_available_strategies()

    for strategy in strategies:
        # Set environment variable
        monkeypatch.setenv("VLLM_MOE_ROUTING_STRATEGY", strategy)

        # Force reload of environment variable
        envs.environment_variables[
            "VLLM_MOE_ROUTING_STRATEGY"] = lambda s=strategy: s

        # Test the select_experts method
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=False,
            renormalize=True,
            indices_type=torch.long)

        # Verify output shapes
        assert topk_weights.shape == (
            num_tokens, top_k), f"Wrong weights shape for {strategy}"
        assert topk_ids.shape == (num_tokens,
                                  top_k), f"Wrong ids shape for {strategy}"

        # Verify expert IDs are valid
        assert topk_ids.min(
        ) >= 0, f"Invalid expert ID (negative) for {strategy}"
        assert topk_ids.max(
        ) < num_experts, f"Invalid expert ID (too large) for {strategy}"


def test_direct_routing_simulator_methods(device):
    """Test the new routing methods directly."""
    pytest.importorskip("vllm.model_executor.layers.fused_moe.layer")

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Test parameters
    num_tokens = 24
    hidden_size = 12
    num_experts = 4
    top_k = 2

    # Create test data
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Test the new select_experts_with_simulated_strategy method
    strategies = ["uniform_random", "weighted_random"]

    for strategy in strategies:
        topk_weights, topk_ids = \
            FusedMoE.select_experts_with_simulated_strategy(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=top_k,
                strategy=strategy,
                indices_type=torch.long)

        # Verify output shapes
        assert topk_weights.shape == (
            num_tokens, top_k), f"Wrong weights shape for {strategy}"
        assert topk_ids.shape == (num_tokens,
                                  top_k), f"Wrong ids shape for {strategy}"


def test_static_methods():
    """Test that static methods work without instantiation."""
    # Test that we can register strategies without creating an instance
    from vllm.model_executor.layers.fused_moe.routing_simulator import (
        UniformRandomRouting)

    # Register a custom strategy
    custom_strategy = UniformRandomRouting()
    name = "custom_uniform"
    RoutingSimulator.register_strategy(name, custom_strategy)

    # Verify it was registered
    available_strategies = RoutingSimulator.get_available_strategies()
    assert name in available_strategies

    # Test that we can use the static simulate_routing method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_states = torch.randn(10, 8, device=device)
    router_logits = torch.randn(10, 4, device=device)

    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name=name,
        top_k=2)

    assert topk_weights.shape == (10, 2)
    assert topk_ids.shape == (10, 2)


def test_instance_compatibility():
    """Test that static methods work correctly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test static method directly
    hidden_states = torch.randn(10, 8, device=device)
    router_logits = torch.randn(10, 4, device=device)

    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name="uniform_random",
        top_k=2)

    assert topk_weights.shape == (10, 2)
    assert topk_ids.shape == (10, 2)
