# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
from vllm.distributed.eplb.rebalance_execute import (
    AsyncEplbCycleComplete,
    AsyncEplbLayerResult,
)


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
) -> MagicMock:
    """Build a minimal EplbModelState mock with only the three map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    return state


def test_commit_eplb_maps_shape_change():
    """
    The normal path copies the physical_to_logical map in-place. When the number of
    physical experts changes, the old map should be replaced entirely.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
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
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """
    Test that logical_to_physical_map is padded with -1 to fill the
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


def test_move_to_workspace_completes_empty_cycle(monkeypatch: pytest.MonkeyPatch):
    consumed_event = MagicMock()
    model_state = MagicMock(
        pending_result=AsyncEplbCycleComplete(consumed_event=consumed_event),
        rebalanced=True,
    )
    move_from_buffer = MagicMock()
    commit_maps = MagicMock()
    monkeypatch.setattr(eplb_state, "move_from_buffer", move_from_buffer)
    monkeypatch.setattr(eplb_state, "_commit_eplb_maps_for_layer", commit_maps)

    eplb_state._move_to_workspace(model_state, ep_rank=0)

    assert not model_state.rebalanced
    assert model_state.pending_result is None
    move_from_buffer.assert_not_called()
    commit_maps.assert_not_called()
    consumed_event.record.assert_called_once_with()


def test_move_to_workspace_completes_on_last_changed_layer(
    monkeypatch: pytest.MonkeyPatch,
):
    consumed_event = MagicMock()
    result = AsyncEplbLayerResult(
        layer_idx=1,
        new_physical_to_logical_map=torch.tensor([1, 0]),
        transfer_metadata=MagicMock(),
        completes_cycle=True,
        consumed_event=consumed_event,
    )
    model_state = MagicMock(
        expert_buffer=MagicMock(),
        model=MagicMock(expert_weights=[MagicMock(), MagicMock(), MagicMock()]),
        pending_result=result,
        rebalanced=True,
    )
    move_from_buffer = MagicMock()
    commit_maps = MagicMock()
    monkeypatch.setattr(eplb_state, "move_from_buffer", move_from_buffer)
    monkeypatch.setattr(eplb_state, "_commit_eplb_maps_for_layer", commit_maps)

    eplb_state._move_to_workspace(model_state, ep_rank=0)

    assert not model_state.rebalanced
    assert model_state.pending_result is None
    move_from_buffer.assert_called_once()
    commit_maps.assert_called_once_with(
        model_state,
        new_physical_to_logical_map=result.new_physical_to_logical_map,
        layer=1,
    )
    consumed_event.record.assert_called_once_with()


def test_drain_async_uses_explicit_cycle_completion():
    consumed_event = MagicMock()
    result = AsyncEplbLayerResult(
        layer_idx=1,
        new_physical_to_logical_map=torch.tensor([1, 0]),
        transfer_metadata=MagicMock(),
        completes_cycle=True,
        consumed_event=consumed_event,
    )
    model_state = MagicMock(pending_result=result, rebalanced=True)
    state = MagicMock(
        is_async=True,
        model_states={"model": model_state},
    )
    state._all_ranks_result_ready.return_value = True

    EplbState.drain_async(state)

    assert not model_state.rebalanced
    assert model_state.pending_result is None
    consumed_event.record.assert_called_once_with()
