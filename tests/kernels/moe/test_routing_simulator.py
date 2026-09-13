#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for the token-to-expert routing simulator.

This script demonstrates how to use the routing simulator to test
different routing strategies and analyze their performance, including
integration tests with FusedMoEFactory layer.
"""

import tempfile

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    DistributionBasedRouting,
    RoutingSimulator,
)


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
        assert topk_ids.min() >= 0, f"Invalid expert ID (negative) for {strategy}"
        assert topk_ids.max() < num_experts, (
            f"Invalid expert ID (too large) for {strategy}"
        )


def test_uniform_strategy_generates_monolithic_logits(device):
    router_logits = torch.full(
        (16, 32),
        7.0,
        dtype=torch.bfloat16,
        device=device,
    )
    original_logits = router_logits.clone()

    simulated_logits = RoutingSimulator.simulate_monolithic_logits(
        router_logits=router_logits,
        strategy_name="uniform_random",
    )

    assert simulated_logits.shape == router_logits.shape
    assert simulated_logits.device == router_logits.device
    assert simulated_logits.dtype == router_logits.dtype
    assert torch.all((simulated_logits >= 0) & (simulated_logits < 1))
    assert torch.equal(router_logits, original_logits)


def test_normal_strategy_does_not_generate_monolithic_logits(device):
    router_logits = torch.zeros((16, 32), device=device)

    with pytest.raises(NotImplementedError):
        RoutingSimulator.simulate_monolithic_logits(
            router_logits=router_logits,
            strategy_name="normal_routing",
        )


@pytest.mark.parametrize(
    "strategy,expected",
    [("uniform_random", True), ("normal_routing", False)],
)
def test_nvfp4_monolithic_supports_uniform_simulation_only(
    monkeypatch,
    strategy: str,
    expected: bool,
):
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    env_name = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
    monkeypatch.setitem(
        envs.environment_variables,
        env_name,
        lambda: strategy,
    )

    assert (
        TrtLlmNvFp4ExpertsMonolithic._supports_routing_method(
            RoutingMethodType.Simulated,
            weight_key=None,
            activation_key=None,
        )
        is expected
    )


@pytest.mark.parametrize(
    "router_kwargs,expected_method",
    [
        ({}, "RenormalizeNaive"),
        ({"renormalize": False}, "Default"),
        (
            {
                "use_grouped_topk": True,
                "num_expert_group": 4,
                "topk_group": 2,
                "scoring_func": "sigmoid",
                "e_score_correction_bias": torch.zeros(16),
            },
            "DeepSeekV3",
        ),
    ],
)
def test_simulator_preserves_original_routing_method(
    monkeypatch,
    router_kwargs,
    expected_method: str,
):
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.router.router_factory import (
        create_fused_moe_router,
        create_non_simulated_fused_moe_router,
    )

    env_name = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
    monkeypatch.setitem(
        envs.environment_variables,
        env_name,
        lambda: "uniform_random",
    )

    non_simulated_router = create_non_simulated_fused_moe_router(
        top_k=2,
        global_num_experts=16,
        **router_kwargs,
    )
    router = create_fused_moe_router(
        top_k=2,
        global_num_experts=16,
        **router_kwargs,
    )

    assert non_simulated_router.routing_method_type == RoutingMethodType[
        expected_method
    ]
    assert router.routing_method_type == RoutingMethodType.Simulated
    assert router.original_routing_method_type == RoutingMethodType[expected_method]


def test_routing_strategy_integration(monkeypatch, device):
    """Test that the routing strategy environment variable works with
    FusedMoEFactory."""
    pytest.importorskip("vllm.model_executor.layers.fused_moe.layer")

    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

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

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        for strategy in strategies:
            fused_moe = FusedMoEFactory(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=0,
                use_grouped_topk=False,
                renormalize=True,
                prefix=strategy,
            )

            # Set environment variable
            env_name = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
            monkeypatch.setenv(env_name, strategy)

            # Temporarily override the envs lookup so the router factory
            # reads the monkeypatched value instead of the module-load-time
            # default. Use monkeypatch.setitem so the original lambda is
            # restored automatically at teardown.
            monkeypatch.setitem(
                envs.environment_variables,
                env_name,
                lambda s=strategy: s,
            )

            # Test the select_experts method
            topk_weights, topk_ids = fused_moe.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            # Verify output shapes
            assert topk_weights.shape == (num_tokens, top_k), (
                f"Wrong weights shape for {strategy}"
            )
            assert topk_ids.shape == (num_tokens, top_k), (
                f"Wrong ids shape for {strategy}"
            )

            # Verify expert IDs are valid
            assert topk_ids.min() >= 0, f"Invalid expert ID (negative) for {strategy}"
            assert topk_ids.max() < num_experts, (
                f"Invalid expert ID (too large) for {strategy}"
            )


def test_distribution_based_routing_with_custom_strategy():
    """Test registering and using DistributionBasedRouting with custom
    parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Register custom distribution-based strategy
    custom_strategy = DistributionBasedRouting(distribution="normal", mean=2.0, std=0.5)
    RoutingSimulator.register_strategy("custom_normal", custom_strategy)

    # Test data
    num_tokens = 60
    hidden_size = 48
    num_experts = 6
    top_k = 3

    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Use the custom strategy
    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name="custom_normal",
        top_k=top_k,
    )

    # Check output shapes
    assert topk_weights.shape == (num_tokens, top_k)
    assert topk_ids.shape == (num_tokens, top_k)

    # Check that expert IDs are valid
    assert topk_ids.min() >= 0
    assert topk_ids.max() < num_experts


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
        top_k=2,
    )

    assert topk_weights.shape == (10, 2)
    assert topk_ids.shape == (10, 2)
