#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for the token-to-expert routing simulator.

This script demonstrates how to use the routing simulator to test
different routing strategies and analyze their performance, including
integration tests with FusedMoEFactory layer.
"""

import tempfile

import pytest
import torch

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    DistributionBasedRouting,
    RoutingSimulator,
)

BUILTIN_STRATEGIES = [
    strategy
    for strategy in RoutingSimulator.get_available_strategies()
    if strategy != "uniform_subset"
]


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _simulate(
    strategy_name: str,
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(num_tokens, 8, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name=strategy_name,
        top_k=top_k,
    )
    assert topk_weights.shape == (num_tokens, top_k)
    assert topk_ids.shape == (num_tokens, top_k)
    assert topk_ids.min() >= 0
    assert topk_ids.max() < num_experts
    return topk_weights, topk_ids


@pytest.mark.parametrize("strategy", BUILTIN_STRATEGIES)
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("top_k", [1, 4])
def test_builtin_strategies(strategy, num_tokens, top_k, device):
    _simulate(
        strategy,
        num_tokens=num_tokens,
        num_experts=32,
        top_k=top_k,
        device=device,
    )


def test_register_custom_strategy(device):
    RoutingSimulator.register_strategy(
        "custom_normal",
        DistributionBasedRouting(distribution="normal", mean=2.0, std=0.5),
    )
    topk_weights, topk_ids = _simulate(
        "custom_normal",
        num_tokens=16,
        num_experts=8,
        top_k=2,
        device=device,
    )
    assert topk_weights.shape == (16, 2)
    assert topk_ids.shape == (16, 2)


def test_uniform_subset_routing(device):
    RoutingSimulator.register_strategy(
        "test_uniform_subset",
        DistributionBasedRouting(distribution="uniform_subset", subset_size=8),
    )
    _, topk_ids = _simulate(
        "test_uniform_subset",
        num_tokens=16,
        num_experts=32,
        top_k=4,
        device=device,
    )
    assert topk_ids.max() < 8


def test_uniform_subset_requires_subset_size():
    with pytest.raises(ValueError, match="subset_size"):
        DistributionBasedRouting(distribution="uniform_subset")


def test_uniform_subset_rejects_invalid_sizes(device):
    strategy = DistributionBasedRouting(
        distribution="uniform_subset",
        subset_size=4,
    )
    hidden_states = torch.randn(4, 8, device=device)
    router_logits = torch.randn(4, 8, device=device)

    with pytest.raises(ValueError, match="top_k"):
        strategy.route_tokens(hidden_states, router_logits, top_k=8)

    with pytest.raises(ValueError, match="subset_size"):
        strategy.route_tokens(
            hidden_states,
            torch.randn(4, 2, device=device),
            top_k=1,
        )


def test_uniform_subset_env_resolution(device, monkeypatch):
    monkeypatch.setitem(
        envs.environment_variables,
        "VLLM_MOE_ROUTING_SIMULATION_SUBSET_SIZE",
        lambda: 4,
    )
    _, topk_ids = _simulate(
        "uniform_subset",
        num_tokens=8,
        num_experts=10,
        top_k=2,
        device=device,
    )
    assert topk_ids.max() < 4


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FusedMoEFactory integration requires a CUDA MoE backend",
)
def test_routing_strategy_integration(monkeypatch, device):
    pytest.importorskip("vllm.model_executor.layers.fused_moe.layer")
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

    num_tokens = 32
    hidden_size = 16
    num_experts = 4
    top_k = 2
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

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

        for strategy in BUILTIN_STRATEGIES:
            fused_moe = FusedMoEFactory(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=0,
                use_grouped_topk=False,
                renormalize=True,
                prefix=strategy,
            )

            env_name = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
            monkeypatch.setenv(env_name, strategy)
            monkeypatch.setitem(
                envs.environment_variables,
                env_name,
                lambda s=strategy: s,
            )

            topk_weights, topk_ids = fused_moe.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            assert topk_weights.shape == (num_tokens, top_k)
            assert topk_ids.shape == (num_tokens, top_k)
            assert topk_ids.min() >= 0
            assert topk_ids.max() < num_experts
