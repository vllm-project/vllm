# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    physical_to_logical_map = torch.arange(num_experts).unsqueeze(0).expand(
        num_layers, -1
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
        health_latency_threshold=config.health_latency_threshold,
        health_penalty_factor=config.health_penalty_factor,
    )


def test_basic_health_detection():
    """Test that unhealthy experts are detected when latency exceeds threshold."""

    # Setup
    num_layers = 2
    num_experts = 8
    window_size = 100
    threshold = 3.0

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_latency_threshold=threshold,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Simulate normal operation (all experts healthy)
    for step in range(50):
        # All experts active with similar latency (10ms)
        state.expert_latency_window[step, :, :] = 10.0

    state.expert_load_window_step = 50

    # Test with normal latency
    current_latency = torch.full((num_layers, num_experts), 10.0)
    state._update_health_mask(current_latency, log_stats=False)

    # All experts should be healthy
    assert (
        state.expert_health_mask.all()
    ), "All experts should be healthy with normal latency"

    # Inject failure: Expert 3 in layer 0 suddenly slow (100ms)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    current_latency[0, 3] = 100.0
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 3 should be unhealthy (100 > 3.0 * 10)
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

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_latency_threshold=3.0,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Simulate sparse activation pattern
    # Expert 0: Active in 15/20 passes with 10ms
    # Expert 1: Active in 5/20 passes with 12ms
    # Expert 2: Active in 15/20 passes with 10ms, then becomes inactive

    for step in range(15):
        state.expert_latency_window[step, 0, 0] = 10.0  # Expert 0 active
        state.expert_latency_window[step, 0, 1] = 0.0  # Expert 1 inactive
        state.expert_latency_window[step, 0, 2] = 10.0  # Expert 2 active

    for step in range(15, 20):
        state.expert_latency_window[step, 0, 0] = 0.0  # Expert 0 inactive
        state.expert_latency_window[step, 0, 1] = 12.0  # Expert 1 active
        state.expert_latency_window[step, 0, 2] = 10.0  # Expert 2 active

    state.expert_load_window_step = 20

    # Test with Expert 2 slow (50ms)
    current_latency = torch.zeros(num_layers, num_experts)
    current_latency[0, 0] = 10.0
    current_latency[0, 2] = 50.0  # Expert 2 slow!
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 2 should be unhealthy (50 > 3 * 10, and has 20 activations)
    assert (
        not state.expert_health_mask[0, 2]
    ), "Expert 2 should be unhealthy after spike"

    # Write to window
    state.expert_latency_window[20, 0, 0] = 10.0
    state.expert_latency_window[20, 0, 2] = 50.0

    # Step 20: Expert 2 becomes inactive (0 latency)
    current_latency = torch.zeros(num_layers, num_experts)
    current_latency[0, 0] = 10.0  # Expert 0 active
    current_latency[0, 2] = 0.0  # Expert 2 NOW INACTIVE (key test!)

    state.expert_load_window_step = 21
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 2 should become healthy again (current_latency = 0 means not active)
    # The check is: (current_latency > 0) & (current_latency > threshold)
    # Since current_latency = 0, first condition is False → not unhealthy
    assert state.expert_health_mask[0, 2], (
        "Expert 2 should be healthy when inactive (current=0), "
        "even though it has >10 activations and was unhealthy before"
    )

    # Expert 0: mean = 10ms, active 16 times (>10), current = 10ms → healthy
    assert state.expert_health_mask[0, 0], "Expert 0 should be healthy"

    # Expert 1: active only 5 times (<10), insufficient history → healthy
    assert state.expert_health_mask[0, 1], (
        "Expert 1 should be healthy (insufficient history)"
    )


def test_expert_recovery():
    """Test that experts can recover from unhealthy state."""

    # Setup
    num_layers = 1
    num_experts = 4
    window_size = 50

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_latency_threshold=3.0,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Phase 1: Normal operation (10ms baseline)
    for step in range(30):
        state.expert_latency_window[step, 0, :] = 10.0

    state.expert_load_window_step = 30
    
    current_latency = torch.full((num_layers, num_experts), 10.0)
    state._update_health_mask(current_latency, log_stats=False)
    assert state.expert_health_mask.all(), "All healthy initially"

    # Write to window
    state.expert_latency_window[30, 0, :] = 10.0

    # Phase 2: Expert 1 becomes slow (50ms)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    current_latency[0, 1] = 50.0
    state.expert_load_window_step = 31
    state._update_health_mask(current_latency, log_stats=False)

    assert not state.expert_health_mask[0, 1], "Expert 1 should be unhealthy"

    # Write to window
    state.expert_latency_window[31, 0, :] = 10.0
    state.expert_latency_window[31, 0, 1] = 50.0

    # Phase 3: Expert 1 recovers immediately (back to 10ms in next step)
    current_latency = torch.full((num_layers, num_experts), 10.0)
    state.expert_load_window_step = 32
    state._update_health_mask(current_latency, log_stats=False)

    # Expert 1 should recover immediately
    # historical_mean now includes the 50ms spike in window
    # current = 10ms should be < threshold → healthy
    assert state.expert_health_mask[0, 1], "Expert 1 should recover to healthy"


def test_multi_layer_health():
    """Test that health detection works independently per layer."""

    num_layers = 3
    num_experts = 8
    window_size = 20

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_latency_threshold=3.0,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Setup baseline (10ms for all)
    for step in range(15):
        state.expert_latency_window[step, :, :] = 10.0

    state.expert_load_window_step = 15

    # Inject failures in different layers
    current_latency = torch.full((num_layers, num_experts), 10.0)
    current_latency[0, 2] = 50.0  # Layer 0, Expert 2: slow
    current_latency[1, 5] = 60.0  # Layer 1, Expert 5: slow
    # Layer 2: all healthy (10.0)

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


def test_historical_mean_calculation():
    """Test that historical mean is calculated correctly with sparse activations."""

    num_layers = 1
    num_experts = 4
    window_size = 10

    eplb_config = EPLBConfig(
        window_size=window_size,
        health_check_enabled=True,
        health_latency_threshold=2.0,
    )

    state = create_mock_eplb_state(
        num_layers=num_layers,
        num_experts=num_experts,
        config=eplb_config,
    )

    # Expert 0 active pattern: Need min(10, 20*0.5) = 10 active passes
    # Pattern: [10, 10, 10, 15, 15, 20, 20, 20, 12, 12, ...]
    # Mean should be (10*3 + 15*2 + 20*3 + 12*2) / 10 = 154 / 10 = 15.4ms

    active_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    latencies = [10.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0, 20.0, 12.0, 12.0]

    for i, step in enumerate(active_steps):
        state.expert_latency_window[step, 0, 0] = latencies[i]

    state.expert_load_window_step = 10

    # Test current latency 30.9ms against historical (steps 0-9)
    # Historical mean = (10*3 + 15*2 + 20*3 + 12*2) / 10 = 154/10 = 15.4ms
    # threshold = 2 * 15.4 = 30.8ms
    # current = 30.9ms > 30.8ms → unhealthy
    current_latency = torch.zeros(1, num_experts, dtype=torch.float32)
    current_latency[0, 0] = 30.9
    state._update_health_mask(current_latency, log_stats=False)

    assert (
        not state.expert_health_mask[0, 0]
    ), "Expert should be unhealthy (30.9 > 30.8)"

    # Now write the 30.9 to window and test next step
    state.expert_latency_window[10 % window_size, 0, 0] = 30.9
    state.expert_load_window_step = 11

    # Test current latency 31ms against historical
    # (now includes 30.9 at step 0)
    # Historical = steps 1-9 + step 0(30.9)
    # = (10*2 + 15*2 + 20*3 + 12*2 + 30.9) / 10 = 174.9/10 = 17.49ms
    # threshold = 2 * 17.49 = 34.98ms
    # current = 31ms < 34.98ms → healthy
    current_latency[0, 0] = 31.0
    state._update_health_mask(current_latency, log_stats=False)

    assert state.expert_health_mask[0, 0], "Expert should be healthy (31 < 34.98)"

    # Test current latency 35ms against historical
    current_latency[0, 0] = 35.0
    state._update_health_mask(current_latency, log_stats=False)

    assert not state.expert_health_mask[0, 0], "Expert should be unhealthy (35 > 34.98)"


