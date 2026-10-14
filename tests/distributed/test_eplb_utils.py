# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.eplb.eplb_state import (
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
    _record_step_expert_load,
)


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    phy2log_storage: torch.Tensor | None = None,
) -> MagicMock:
    """Build a minimal EplbModelState mock with its map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.physical_to_logical_map_buffer = (
        phy2log if phy2log_storage is None else phy2log_storage
    )
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    return state


def test_commit_eplb_maps_shape_change():
    """When the number of physical experts changes, resize the active map within its
    preallocated storage.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    storage = torch.full((num_layers, num_physical + 2), -1, dtype=torch.long)
    model_state = _make_model_state(
        phy2log=storage[:, :num_physical],
        phy2log_storage=storage,
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    # The new map has two more physical experts. These new physical experts will
    # automatically map to the first two logical experts
    new_phy2log_larger = (
        (torch.arange(num_physical + 2, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
    )
    _commit_eplb_maps(model_state, new_phy2log_larger)

    # Check that the number of physical experts has been updated and that the values
    # match
    assert model_state.physical_to_logical_map.shape[1] == num_physical + 2
    assert model_state.physical_to_logical_map.data_ptr() == storage.data_ptr()
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """Test that logical_to_physical_map is padded with -1 to fill the
    pre-allocated slots when the new map has fewer replicas than the max.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = (
        (torch.arange(num_physical, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    layer = 0
    _commit_eplb_maps_for_layer(model_state, new_phy2log[layer], layer)

    assert torch.all(model_state.logical_to_physical_map[layer, :, 2] == -1)


def test_commit_eplb_maps_for_layer_shape_assert():
    """Test that a mismatched number of physical experts triggers an assertion error."""
    num_layers, num_logical, num_physical = 2, 4, 6

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full((num_layers, num_logical, 2), -1, dtype=torch.long),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )
    bad_phy2log = torch.zeros(num_layers, num_physical + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        _commit_eplb_maps_for_layer(model_state, bad_phy2log, layer=0)


def test_commit_eplb_maps():
    """Test that all values are copied correctly into model_state."""
    num_layers, num_logical, num_physical, max_replicas = 2, 3, 4, 2

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps(model_state, new_phy2log)

    assert torch.equal(model_state.physical_to_logical_map, new_phy2log)
    assert torch.equal(model_state.logical_to_physical_map, new_log2phy)
    assert torch.equal(model_state.logical_replica_count, new_logcnt)


def test_commit_eplb_maps_for_layer():
    """Test that only the target layer is updated"""
    num_layers, num_logical, max_replicas = 2, 3, 2

    original_phy2log = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    model_state = _make_model_state(
        phy2log=original_phy2log.clone(),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps_for_layer(model_state, new_phy2log[0], layer=0)

    # Layer 0 updated
    assert torch.equal(model_state.physical_to_logical_map[0], new_phy2log[0])
    assert torch.equal(model_state.logical_to_physical_map[0], new_log2phy[0])
    assert torch.equal(model_state.logical_replica_count[0], new_logcnt[0])

    # Layer 1 untouched
    assert torch.equal(model_state.physical_to_logical_map[1], original_phy2log[1])


def test_record_step_expert_load():
    """Verify that _record_step_expert_load maps physical loads into logical expert
    slots at step time and ignores unmapped physical slots."""
    num_layers, num_logical = 2, 4
    window_size = 3

    # Layer 0: experts 0, 1, 2, 3, 0, 1 (0 and 1 replicated)
    # Layer 1: experts 2, 3, 0, 1, -1, -1 (unmapped slots with -1)
    phy2log = torch.tensor([[0, 1, 2, 3, 0, 1], [2, 3, 0, 1, -1, -1]], dtype=torch.long)
    expert_load_pass = torch.tensor(
        [[10, 20, 30, 40, 5, 15], [7, 8, 9, 10, 999, 888]], dtype=torch.int32
    )
    expert_load_window = torch.zeros(
        (window_size, num_layers, num_logical), dtype=torch.int32
    )
    logical_load_buffer = torch.zeros((num_layers, num_logical + 1), dtype=torch.int32)

    model_state = MagicMock()
    model_state.model.num_logical_experts = num_logical
    model_state.model.num_moe_layers = num_layers
    model_state.physical_to_logical_map = phy2log
    model_state.expert_load_pass = expert_load_pass
    model_state.expert_load_window = expert_load_window
    model_state.logical_load_buffer = logical_load_buffer

    # Record at window step 0
    _record_step_expert_load(model_state, window_step=0)

    # Layer 0: expert 0 has 10 + 5 = 15; expert 1 has 20 + 15 = 35;
    # expert 2 has 30; expert 3 has 40
    expected_layer_0 = torch.tensor([15, 35, 30, 40], dtype=torch.int32)
    # Layer 1: expert 2 has 7; expert 3 has 8; expert 0 has 9; expert 1 has 10;
    # unmapped slots discarded
    expected_layer_1 = torch.tensor([9, 10, 7, 8], dtype=torch.int32)

    assert torch.equal(expert_load_window[0, 0], expected_layer_0)
    assert torch.equal(expert_load_window[0, 1], expected_layer_1)
    # Ensure pass accumulator was zeroed
    assert torch.all(model_state.expert_load_pass == 0)


def test_expert_load_window_historical_mapping_across_rearrangement():
    """Verify issue #60685: when step_interval < window_size and rearrangement
    swaps expert physical locations, past step loads in the sliding window are not
    falsely reattributed to the newly placed expert."""
    num_layers, num_logical = 1, 2
    window_size = 2

    # Step 0: slot 0 -> logical 0, slot 1 -> logical 1
    phy2log = torch.tensor([[0, 1]], dtype=torch.long)
    expert_load_pass = torch.tensor([[10, 0]], dtype=torch.int32)
    expert_load_window = torch.zeros(
        (window_size, num_layers, num_logical), dtype=torch.int32
    )

    model_state = MagicMock()
    model_state.model.num_logical_experts = num_logical
    model_state.model.num_moe_layers = num_layers
    model_state.physical_to_logical_map = phy2log
    model_state.expert_load_pass = expert_load_pass
    model_state.expert_load_window = expert_load_window
    model_state.logical_load_buffer = None

    # Step 0: Logical expert 0 processes 10 tokens
    _record_step_expert_load(model_state, window_step=0)

    # Rearrangement swaps experts: slot 0 -> logical 1, slot 1 -> logical 0
    new_phy2log = torch.tensor([[1, 0]], dtype=torch.long)
    model_state.physical_to_logical_map = new_phy2log

    # Step 1: Physical slot 0 processes 20 tokens (which is now logical expert 1)
    model_state.expert_load_pass = torch.tensor([[20, 0]], dtype=torch.int32)
    _record_step_expert_load(model_state, window_step=1)

    # Aggregating the sliding window across both steps
    total_window_load = expert_load_window.sum(dim=0)

    # In the bug (#60685), step 0's slot 0 load (10) was reattributed using
    # the new map [1, 0], falsely assigning all 30 tokens to logical expert 1
    # and 0 tokens to logical expert 0.
    # With the fix, step 0 correctly recorded load for expert 0, and step 1
    # for expert 1.
    assert total_window_load[0, 0].item() == 10  # Logical expert 0
    assert total_window_load[0, 1].item() == 20  # Logical expert 1
