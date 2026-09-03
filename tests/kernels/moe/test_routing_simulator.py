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


# ---------------------------------------------------------------------------
# Per-step batched RNG (one rand+topk per forward pass shared across layers)
# ---------------------------------------------------------------------------

from vllm.forward_context import ForwardContext, override_forward_context  # noqa: E402


def _fresh_forward_context() -> ForwardContext:
    return ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})


def _run_pass(
    strategy: DistributionBasedRouting,
    num_layers: int,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Simulate one forward pass: num_layers routed calls under one context."""
    with override_forward_context(_fresh_forward_context()):
        return [
            strategy.route_tokens(hidden_states, router_logits, top_k)
            for _ in range(num_layers)
        ]


def _check_valid(ids: torch.Tensor, num_tokens: int, num_experts: int, top_k: int):
    assert ids.shape == (num_tokens, top_k)
    assert ids.min() >= 0
    assert ids.max() < num_experts
    # Without-replacement: each token's top-k experts are distinct.
    for row in ids.tolist():
        assert len(set(row)) == top_k


def test_step_batched_engages_after_learning_pass():
    device = torch.device("cpu")
    num_layers, num_tokens, num_experts, top_k = 5, 4, 32, 3
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(num_tokens, 8, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Pass 1 learns the layer count via the per-call path.
    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    assert strategy.step_batch_stats["batched_calls"] == 0

    # Pass 2 serves every layer from one batched draw.
    outs = _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    assert strategy.step_batch_stats["batched_calls"] == num_layers
    for weights, ids in outs:
        _check_valid(ids, num_tokens, num_experts, top_k)
        assert ids.dtype == torch.long
        assert weights.shape == (num_tokens, top_k)
        assert torch.equal(weights, torch.ones_like(weights))
    # One cached ones-weights buffer shared across the step's layers.
    assert all(out[0] is outs[0][0] for out in outs)


def test_step_batched_determinism_per_step():
    """Invariant: fixed generator state at pass start => bitwise-identical
    routing for every layer of the pass."""
    device = torch.device("cpu")
    num_layers, num_tokens, num_experts, top_k = 6, 2, 16, 4
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(num_tokens, 8, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    base = strategy.step_batch_stats["batched_calls"]

    torch.manual_seed(1234)
    pass_a = _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    torch.manual_seed(1234)
    pass_b = _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)

    assert strategy.step_batch_stats["batched_calls"] - base == 2 * num_layers
    for (w_a, ids_a), (w_b, ids_b) in zip(pass_a, pass_b):
        assert torch.equal(ids_a, ids_b)
        assert torch.equal(w_a, w_b)


def test_step_batched_ids_vary_across_steps_and_layers():
    device = torch.device("cpu")
    num_layers, num_tokens, num_experts, top_k = 5, 8, 64, 4
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(num_tokens, 8, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    steps = [
        [
            ids.clone()
            for _, ids in _run_pass(
                strategy, num_layers, hidden_states, router_logits, top_k
            )
        ]
        for _ in range(10)
    ]
    # Ids vary across consecutive steps.
    assert any(
        not torch.equal(steps[i][0], steps[i + 1][0]) for i in range(len(steps) - 1)
    )
    # Layers within one step are decorrelated (slices of one batch).
    first = steps[0]
    assert any(not torch.equal(first[0], first[i]) for i in range(1, num_layers))


def test_step_batched_prefill_falls_back_but_learns():
    device = torch.device("cpu")
    num_layers, num_experts, top_k = 4, 32, 2
    strategy = DistributionBasedRouting(distribution="uniform")
    logits_big = torch.randn(128, num_experts, device=device)
    hidden_big = torch.randn(128, 8, device=device)

    # Prefill-sized pass: per-call path for every layer.
    outs = _run_pass(strategy, num_layers, hidden_big, logits_big, top_k)
    assert strategy.step_batch_stats["batched_calls"] == 0
    assert strategy.step_batch_stats["fallback_calls"] == num_layers
    for weights, ids in outs:
        _check_valid(ids, 128, num_experts, top_k)
        assert torch.equal(weights, torch.ones_like(weights))

    # ...but the layer count was learned: the first decode-sized pass batches.
    hidden_small = torch.randn(2, 8, device=device)
    logits_small = torch.randn(2, num_experts, device=device)
    _run_pass(strategy, num_layers, hidden_small, logits_small, top_k)
    assert strategy.step_batch_stats["batched_calls"] == num_layers


def test_step_batched_overflow_fails_closed_then_relearns():
    device = torch.device("cpu")
    num_experts, top_k = 16, 2
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(4, 8, device=device)
    router_logits = torch.randn(4, num_experts, device=device)

    _run_pass(strategy, 3, hidden_states, router_logits, top_k)  # learn 3
    fallbacks_after_learning = strategy.step_batch_stats["fallback_calls"]
    outs = _run_pass(strategy, 5, hidden_states, router_logits, top_k)
    assert strategy.step_batch_stats["batched_calls"] == 3
    assert strategy.step_batch_stats["fallback_calls"] == fallbacks_after_learning + 2
    for weights, ids in outs:
        _check_valid(ids, 4, num_experts, top_k)

    # The stale count was dropped and relearned from the overflowing pass.
    _run_pass(strategy, 5, hidden_states, router_logits, top_k)
    assert strategy.step_batch_stats["batched_calls"] == 3 + 5


def test_step_batched_requires_forward_context():
    device = torch.device("cpu")
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(4, 8, device=device)
    router_logits = torch.randn(4, 16, device=device)
    outs = [strategy.route_tokens(hidden_states, router_logits, 2) for _ in range(4)]
    assert strategy.step_batch_stats["batched_calls"] == 0
    # Per-call path allocates fresh ones-weights (no shared buffer).
    assert outs[0][0] is not outs[1][0]
    for weights, ids in outs:
        _check_valid(ids, 4, 16, 2)
        assert torch.equal(weights, torch.ones_like(weights))


def test_step_batched_dbo_falls_back_and_never_learns(monkeypatch):
    """DBO detection must key off the ubatch thread registry: the
    per-ubatch ForwardContexts are created without ubatch_slices, so a
    context-attribute check can never fire inside ubatch threads."""
    import threading

    from vllm.v1.worker import ubatching

    device = torch.device("cpu")
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(4, 8, device=device)
    router_logits = torch.randn(4, 16, device=device)

    # Simulate running inside a DBO ubatch thread.
    monkeypatch.setitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), 0)
    assert ubatching.dbo_enabled()

    # Two lock-step ubatch threads alternate contexts (as under DBO,
    # where each per-ubatch context lacks ubatch_slices): every call
    # must take the per-call path and nothing may be learned.
    ctx_a, ctx_b = _fresh_forward_context(), _fresh_forward_context()
    for _ in range(3):
        for ctx in (ctx_a, ctx_b):
            with override_forward_context(ctx):
                weights, ids = strategy.route_tokens(hidden_states, router_logits, 2)
                _check_valid(ids, 4, 16, 2)
                assert torch.equal(weights, torch.ones_like(weights))
    assert strategy.step_batch_stats["batched_calls"] == 0
    assert strategy._learned_layers == {}

    # Out of the ubatch thread, learn-then-engage behavior resumes.
    monkeypatch.delitem(ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident())
    assert not ubatching.dbo_enabled()
    _run_pass(strategy, 2, hidden_states, router_logits, 2)
    _run_pass(strategy, 2, hidden_states, router_logits, 2)
    assert strategy.step_batch_stats["batched_calls"] == 2


def test_step_batched_mixed_shapes_never_learn_corrupt_counts():
    device = torch.device("cpu")
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(4, 8, device=device)
    logits_a = torch.randn(4, 16, device=device)
    logits_b = torch.randn(4, 32, device=device)

    # Heterogeneous pass: two layer groups with different expert counts.
    with override_forward_context(_fresh_forward_context()):
        for _ in range(2):
            strategy.route_tokens(hidden_states, logits_a, 2)
        for _ in range(2):
            strategy.route_tokens(hidden_states, logits_b, 2)
    # Nothing learned from the dirty pass; the next pass is per-call.
    _run_pass(strategy, 2, hidden_states, logits_a, 2)
    assert strategy.step_batch_stats["batched_calls"] == 0
    # A homogeneous pass re-enables batching afterwards.
    _run_pass(strategy, 2, hidden_states, logits_a, 2)
    assert strategy.step_batch_stats["batched_calls"] == 2


def test_step_batched_capture_created_ones_filled_before_eager_use(monkeypatch):
    """A ones buffer created during CUDA graph capture has its fill
    recorded, not executed: until the creating graph replays, the cached
    buffer may hold arbitrary pool memory. Every other reader must fill it
    before use -- one recorded fill per other capture, one executed fill on
    the first out-of-capture read (which settles it for good)."""
    import vllm.model_executor.layers.fused_moe.router.routing_simulator_router as rsr

    device = torch.device("cpu")
    num_layers, num_tokens, num_experts, top_k = 3, 4, 16, 2
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(num_tokens, 8, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    key = (num_tokens, top_k, device)

    # Learning pass (eager, per-call): must not touch the cache.
    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    assert key not in strategy._ones_cache

    # 'Capture' pass creates the cache entry: it must be tracked as
    # pending (its real fill would only be recorded, not executed).
    monkeypatch.setattr(rsr, "_is_current_stream_capturing", lambda: True)
    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    assert strategy.step_batch_stats["batched_calls"] == num_layers
    assert key in strategy._ones_cache
    assert key in strategy._ones_pending_fill

    # Model the not-yet-replayed graph-pool state (on CPU the recorded
    # fill executed immediately, so poison the buffer by hand).
    strategy._ones_cache[key].fill_(float("nan"))

    # A second capture baking the same address must record its own fill
    # so its replays never depend on the first graph having replayed.
    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    assert torch.equal(strategy._ones_cache[key], torch.ones((num_tokens, top_k)))
    assert key in strategy._ones_pending_fill

    # First out-of-capture reader: must see all-ones, never pool garbage,
    # and the executed fill clears the pending state.
    strategy._ones_cache[key].fill_(float("nan"))
    monkeypatch.setattr(rsr, "_is_current_stream_capturing", lambda: False)
    outs = _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    for weights, _ in outs:
        assert torch.equal(weights, torch.ones((num_tokens, top_k)))
    assert key not in strategy._ones_pending_fill


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_step_batched_cudagraph_replay_ids_vary():
    """Graph-safe Philox: the in-capture batched draw yields fresh ids per
    replay, and slices stay valid without-replacement rows."""
    device = torch.device("cuda")
    num_layers, num_tokens, num_experts, top_k = 3, 4, 32, 2
    strategy = DistributionBasedRouting(distribution="uniform")
    hidden_states = torch.randn(num_tokens, 8, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Learning pass (eager), then a warmup pass on a side stream.
    _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_pass(strategy, num_layers, hidden_states, router_logits, top_k)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with (
        override_forward_context(_fresh_forward_context()),
        torch.cuda.graph(graph),
    ):
        outs = [
            strategy.route_tokens(hidden_states, router_logits, top_k)
            for _ in range(num_layers)
        ]
    assert strategy.step_batch_stats["batched_calls"] >= 2 * num_layers

    graph.replay()
    torch.accelerator.synchronize()
    snap1 = [ids.clone() for _, ids in outs]
    graph.replay()
    torch.accelerator.synchronize()
    snap2 = [ids.clone() for _, ids in outs]

    assert any(not torch.equal(a, b) for a, b in zip(snap1, snap2))
    for ids in snap2:
        _check_valid(ids.cpu(), num_tokens, num_experts, top_k)
