# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import EPLBConfig
from vllm.distributed.eplb.eplb_state import EplbState


def create_mock_eplb_state(
    num_layers: int,
    num_experts: int,
    config: EPLBConfig,
) -> EplbState:
    """Create a mock EPLB state for testing."""

    # Create minimal required tensors
    physical_to_logical_map = (
        torch.arange(num_experts).unsqueeze(0).expand(num_layers, -1)
    )
    logical_to_physical_map = (
        torch.arange(num_experts).unsqueeze(1).unsqueeze(0).expand(num_layers, -1, -1)
    )
    logical_replica_count = torch.ones(num_layers, num_experts, dtype=torch.long)
    expert_load_pass = torch.zeros(num_layers, num_experts, dtype=torch.int32)
    expert_load_window = torch.zeros(
        config.window_size, num_layers, num_experts, dtype=torch.int32
    )

    # Health monitoring tensors
    expert_latency_window = torch.zeros(
        config.window_size, num_layers, num_experts, dtype=torch.float32
    )
    expert_health_mask = torch.ones(num_layers, num_experts, dtype=torch.bool)

    return EplbState(
        physical_to_logical_map=physical_to_logical_map,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
        expert_load_pass=expert_load_pass,
        expert_load_window=expert_load_window,
        expert_load_window_step=0,
        expert_load_window_size=config.window_size,
        expert_rearrangement_step=0,
        expert_rearrangement_step_interval=config.step_interval,
        expert_latency_window=expert_latency_window,
        expert_health_mask=expert_health_mask,
        health_timeout_threshold=config.health_timeout_threshold,
    )


def test_basic_health_detection():
    """Test that unhealthy experts are detected when exceeding timeout."""

    # Setup
    num_layers = 2
    num_experts = 8
    window_size = 100
    timeout_threshold = 50.0  # 50ms absolute timeout

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_timeout_threshold=timeout_threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Test with normal latency (10ms < 50ms timeout)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    state._update_health_mask(current_latency, log_stats=False)

    # All experts should be healthy
    assert state.expert_health_mask.all(), (
        "All experts should be healthy with normal latency"
    )

    # Inject failure: Expert 3 in layer 0 exceeds timeout (100ms > 50ms)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    current_latency[0, 3] = 100.0
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 3 should be unhealthy (100ms > 50ms timeout)
    assert not state.expert_health_mask[0, 3], "Expert 3 layer 0 should be unhealthy"
    # Other experts should remain healthy
    assert state.expert_health_mask[0, :3].all(), "Other experts should remain healthy"
    assert state.expert_health_mask[0, 4:].all(), "Other experts should remain healthy"
    assert state.expert_health_mask[1, :].all(), "Layer 1 experts should remain healthy"


def test_sparse_activation():
    """Test that inactive experts (0 latency) are handled correctly."""

    # Setup
    num_layers = 1
    num_experts = 8
    window_size = 30
    timeout_threshold = 40.0  # 40ms absolute timeout

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_timeout_threshold=timeout_threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Test with Expert 2 exceeding timeout (50ms > 40ms)
    current_latency = torch.zeros(num_layers, num_experts)
    current_latency[0, 0] = 10.0
    current_latency[0, 2] = 50.0  # Expert 2 slow!
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 2 should be unhealthy (50ms > 40ms timeout)
    assert not state.expert_health_mask[0, 2], (
        "Expert 2 should be unhealthy after exceeding timeout"
    )

    # Expert 2 becomes inactive (0 latency)
    current_latency = torch.zeros(num_layers, num_experts)
    current_latency[0, 0] = 10.0  # Expert 0 active
    current_latency[0, 2] = 0.0  # Expert 2 NOW INACTIVE (key test!)

    state._update_health_mask(current_latency, log_stats=False)

    # Expert 2 should become healthy again (current_latency = 0 means not active)
    # The check is: (current_latency > 0) & (current_latency > timeout_threshold)
    # Since current_latency = 0, first condition is False → not unhealthy
    assert state.expert_health_mask[0, 2], (
        "Expert 2 should be healthy when inactive (current=0)"
    )

    # Expert 0 with 10ms < 40ms timeout → healthy
    assert state.expert_health_mask[0, 0], "Expert 0 should be healthy"

    # Expert 1 inactive (0 latency) → healthy
    assert state.expert_health_mask[0, 1], "Expert 1 should be healthy (inactive)"


def test_expert_recovery():
    """Test that experts can recover from unhealthy state."""

    # Setup
    num_layers = 1
    num_experts = 4
    window_size = 50
    timeout_threshold = 40.0  # 40ms absolute timeout

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_timeout_threshold=timeout_threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Phase 1: Normal operation (10ms < 40ms timeout)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    state._update_health_mask(current_latency, log_stats=False)
    assert state.expert_health_mask.all(), "All healthy initially"

    # Phase 2: Expert 1 exceeds timeout (50ms > 40ms)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    current_latency[0, 1] = 50.0
    state._update_health_mask(current_latency, log_stats=False)

    assert not state.expert_health_mask[0, 1], "Expert 1 should be unhealthy"

    # Phase 3: Expert 1 recovers immediately (back to 10ms < 40ms in next step)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 1 should recover immediately
    # current = 10ms < 40ms timeout → healthy
    assert state.expert_health_mask[0, 1], "Expert 1 should recover to healthy"


def test_multi_layer_health():
    """Test that health detection works independently per layer."""

    num_layers = 3
    num_experts = 8
    window_size = 20
    timeout_threshold = 40.0  # 40ms absolute timeout

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_timeout_threshold=timeout_threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Inject failures in different layers
    current_latency = torch.full((num_layers, num_experts), 10.0)
    current_latency[0, 2] = 50.0  # Layer 0, Expert 2: exceeds timeout (50ms > 40ms)
    current_latency[1, 5] = 60.0  # Layer 1, Expert 5: exceeds timeout (60ms > 40ms)
    # Layer 2: all healthy (10ms < 40ms)

    state._update_health_mask(current_latency, log_stats=False)

    # Verify layer 0
    assert not state.expert_health_mask[0, 2], "Layer 0 Expert 2 should be unhealthy"
    assert state.expert_health_mask[0, :2].all(), "Layer 0 other experts healthy"
    assert state.expert_health_mask[0, 3:].all(), "Layer 0 other experts healthy"

    # Verify layer 1
    assert not state.expert_health_mask[1, 5], "Layer 1 Expert 5 should be unhealthy"
    assert state.expert_health_mask[1, :5].all(), "Layer 1 other experts healthy"
    assert state.expert_health_mask[1, 6:].all(), "Layer 1 other experts healthy"

    # Verify layer 2
    assert state.expert_health_mask[2, :].all(), "Layer 2 all experts should be healthy"


def test_timeout_threshold_boundary():
    """Test timeout threshold boundary conditions."""

    num_layers = 1
    num_experts = 4
    window_size = 10
    timeout_threshold = 30.0  # 30ms absolute timeout

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_timeout_threshold=timeout_threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Test latency just below threshold (29.9ms < 30ms)
    current_latency = torch.zeros(1, num_experts, dtype=torch.float32)
    current_latency[0, 0] = 29.9
    state._update_health_mask(current_latency, log_stats=False)

    assert state.expert_health_mask[0, 0], (
        "Expert should be healthy (29.9ms < 30ms threshold)"
    )

    # Test latency at threshold (30ms = 30ms) - should be healthy (not greater than)
    current_latency[0, 0] = 30.0
    state._update_health_mask(current_latency, log_stats=False)

    assert state.expert_health_mask[0, 0], (
        "Expert should be healthy (30ms = 30ms threshold, not exceeding)"
    )

    # Test latency just above threshold (30.1ms > 30ms)
    current_latency[0, 0] = 30.1
    state._update_health_mask(current_latency, log_stats=False)

    assert not state.expert_health_mask[0, 0], (
        "Expert should be unhealthy (30.1ms > 30ms threshold)"
    )

    # Test latency well above threshold (100ms > 30ms)
    current_latency[0, 0] = 100.0
    state._update_health_mask(current_latency, log_stats=False)

    assert not state.expert_health_mask[0, 0], (
        "Expert should be unhealthy (100ms > 30ms threshold)"
    )

    # Test inactive expert (0ms) even if it was previously unhealthy
    current_latency[0, 0] = 0.0
    state._update_health_mask(current_latency, log_stats=False)

    assert state.expert_health_mask[0, 0], (
        "Inactive expert should be considered healthy"
    )


def test_immediate_masking():
    """Test that unhealthy experts are immediately masked out."""

    num_layers = 1
    num_experts = 4
    window_size = 10
    timeout_threshold = 50.0  # 50ms absolute timeout

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_timeout_threshold=timeout_threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Initial state: all experts are healthy and mapped
    # logical_to_physical_map has 1:1 mapping (no redundancy in mock state)
    assert state.logical_replica_count[0, 0] == 1, "Expert 0 should have 1 replica"
    assert state.logical_to_physical_map[0, 0, 0] == 0, (
        "Expert 0 should map to physical 0"
    )

    # Expert 1 exceeds timeout (100ms > 50ms)
    current_latency = torch.zeros(num_layers, num_experts)
    current_latency[0, 1] = 100.0
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 1 should be immediately masked out
    assert not state.expert_health_mask[0, 1], "Expert 1 should be unhealthy"
    assert state.logical_replica_count[0, 1] == 0, (
        "Expert 1 should have 0 replicas (masked)"
    )
    # The physical expert should be removed from the mapping (set to -1)
    assert state.logical_to_physical_map[0, 1, 0] == -1, (
        "Expert 1 should be masked from mapping"
    )

    # Other experts should remain unaffected
    assert state.expert_health_mask[0, 0], "Expert 0 should be healthy"
    assert state.logical_replica_count[0, 0] == 1, (
        "Expert 0 should still have 1 replica"
    )
    assert state.logical_to_physical_map[0, 0, 0] == 0, "Expert 0 mapping unchanged"

    # Expert 1 recovers (latency drops below threshold)
    current_latency[0, 1] = 10.0
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 1 should be immediately unmasked
    assert state.expert_health_mask[0, 1], "Expert 1 should be healthy again"
    assert state.logical_replica_count[0, 1] == 1, (
        "Expert 1 should have 1 replica again"
    )
    assert state.logical_to_physical_map[0, 1, 0] == 1, "Expert 1 should be remapped"


# =============================================================================
# Tests for deferred latency measurement
# =============================================================================


def test_fusedmoe_has_measure_method():
    """Test that FusedMoE layer has the measure_and_update_latency method."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Create minimal VllmConfig with health monitoring enabled
    vllm_config = VllmConfig()
    vllm_config.parallel_config.enable_eplb = True
    vllm_config.parallel_config.eplb_config.health_check_enabled = True

    with set_current_vllm_config(vllm_config):
        # Create a minimal FusedMoE layer
        # Provide tp_size, ep_size, dp_size to avoid distributed initialization
        layer = FusedMoE(
            num_experts=8,
            top_k=2,
            hidden_size=128,
            intermediate_size=256,
            tp_size=1,
            ep_size=1,
            dp_size=1,
            enable_eplb=True,  # Enable EPLB to trigger health monitoring
        )

        # Verify the method exists
        assert hasattr(layer, "measure_and_update_latency"), (
            "FusedMoE should have measure_and_update_latency method"
        )

        # Verify it's callable
        assert callable(layer.measure_and_update_latency), (
            "measure_and_update_latency should be callable"
        )

        # Verify deferred measurement attributes exist
        assert hasattr(layer, "_pending_active_experts"), (
            "FusedMoE should have _pending_active_experts attribute"
        )
        assert hasattr(layer, "_has_pending_measurement"), (
            "FusedMoE should have _has_pending_measurement attribute"
        )

        # Verify CUDA events are created when health monitoring is enabled
        assert layer.cuda_start_event is not None, (
            "cuda_start_event should be created when health monitoring enabled"
        )
        assert layer.cuda_end_event is not None, (
            "cuda_end_event should be created when health monitoring enabled"
        )

        # Test that measure_and_update_latency doesn't crash when called
        # (even with no pending measurement)
        layer.measure_and_update_latency()  # Should not raise

        # Verify pending state is initially False
        assert not layer._has_pending_measurement, (
            "Should not have pending measurement initially"
        )


def test_fusedmoe_health_monitoring_disabled():
    """Test that FusedMoE doesn't create events when health monitoring is disabled."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Create VllmConfig with health monitoring DISABLED
    vllm_config = VllmConfig()
    vllm_config.parallel_config.enable_eplb = True
    vllm_config.parallel_config.eplb_config.health_check_enabled = False

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=8,
            top_k=2,
            hidden_size=128,
            intermediate_size=256,
            tp_size=1,
            ep_size=1,
            dp_size=1,
            enable_eplb=True,  # Enable EPLB but health monitoring is disabled in config
        )

        # When health monitoring is disabled, events should be None
        assert layer.cuda_start_event is None, (
            "cuda_start_event should be None when health monitoring disabled"
        )
        assert layer.cuda_end_event is None, (
            "cuda_end_event should be None when health monitoring disabled"
        )

        # Method should still exist but be a no-op
        layer.measure_and_update_latency()  # Should not crash


@torch.inference_mode()
def test_fusedmoe_deferred_measurement_state_management():
    """Test that FusedMoE properly manages deferred measurement state."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    device = torch.device("cuda")
    num_experts = 8

    # Create VllmConfig with health monitoring enabled
    vllm_config = VllmConfig()
    vllm_config.parallel_config.enable_eplb = True
    vllm_config.parallel_config.eplb_config.health_check_enabled = True

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=num_experts,
            top_k=2,
            hidden_size=128,
            intermediate_size=256,
            tp_size=1,
            ep_size=1,
            dp_size=1,
            enable_eplb=True,  # Enable EPLB to trigger health monitoring
        )

        # Move layer to CUDA
        layer = layer.to(device)

    # Create latency view tensor
    expert_latency_pass = torch.zeros(1, num_experts, device=device)
    layer.expert_latency_view = expert_latency_pass[0]

    # Simulate setting pending measurement (as forward_impl would)
    layer._pending_active_experts = torch.tensor([0, 2, 5], dtype=torch.long).cpu()
    layer._has_pending_measurement = True

    # Record fake CUDA events
    layer.cuda_start_event.record()
    # Do some work
    dummy = torch.randn(100, 100, device=device)
    for _ in range(5):
        dummy = dummy @ dummy.T
    layer.cuda_end_event.record()

    # Verify pending state is set
    assert layer._has_pending_measurement, "Should have pending measurement"
    assert layer._pending_active_experts is not None

    # Call measure_and_update_latency
    layer.measure_and_update_latency()

    # Verify pending state is cleared
    assert not layer._has_pending_measurement, "Pending flag should be cleared"
    assert layer._pending_active_experts is None, "Pending data should be cleared"

    # Verify latency was recorded for active experts
    assert expert_latency_pass[0, 0] > 0, "Expert 0 should have latency"
    assert expert_latency_pass[0, 2] > 0, "Expert 2 should have latency"
    assert expert_latency_pass[0, 5] > 0, "Expert 5 should have latency"

    # Verify inactive experts have zero latency
    assert expert_latency_pass[0, 1] == 0, "Expert 1 should have zero latency"
    assert expert_latency_pass[0, 3] == 0, "Expert 3 should have zero latency"

    # Test that calling measure again with no pending measurement is a no-op
    layer.measure_and_update_latency()  # Should not crash
