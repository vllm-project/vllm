# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_module
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
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


def test_eplb_config_disable_online_requires_initial_mapping():
    with pytest.raises(ValueError, match="requires initial_mapping_path"):
        EPLBConfig(disable_online_rebalancing=True)


def _make_eplb_state_for_mapping(tmp_path: Path, record: dict) -> EplbState:
    mapping_path = tmp_path / "mapping.jsonl"
    mapping_path.write_text(json.dumps(record) + "\n")
    state = object.__new__(EplbState)
    state.device = torch.device("cpu")
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(initial_mapping_path=str(mapping_path))
    )
    return state


def test_load_initial_mapping_jsonl(tmp_path):
    record = {
        "record_type": "eplb_initial_mapping",
        "version": 1,
        "num_redundant_experts": 1,
        "num_slots": 4,
        "initial_global_assignments": {
            "0": [0, 1, 2, 0],
            "1": [1, 2, 0, 1],
        },
    }
    state = _make_eplb_state_for_mapping(tmp_path, record)

    p2l, l2p, logcnt = state._load_initial_mapping(
        str(tmp_path / "mapping.jsonl"),
        num_moe_layers=2,
        num_physical_experts=4,
        num_logical_experts=3,
        num_redundant_experts=1,
        max_slots_per_logical=4,
    )

    assert torch.equal(p2l.cpu(), torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]))
    assert torch.equal(logcnt.cpu(), torch.tensor([[2, 1, 1], [1, 2, 1]]))
    assert torch.equal(
        l2p.cpu(),
        torch.tensor(
            [
                [[0, 3, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1]],
                [[2, -1, -1, -1], [0, 3, -1, -1], [1, -1, -1, -1]],
            ]
        ),
    )


def test_load_initial_mapping_rejects_redundant_mismatch(tmp_path):
    record = {
        "record_type": "eplb_initial_mapping",
        "version": 1,
        "num_redundant_experts": 2,
        "num_slots": 4,
        "initial_global_assignments": {"0": [0, 1, 2, 0]},
    }
    state = _make_eplb_state_for_mapping(tmp_path, record)

    with pytest.raises(ValueError, match="num_redundant_experts"):
        state._load_initial_mapping(
            str(tmp_path / "mapping.jsonl"),
            num_moe_layers=1,
            num_physical_experts=4,
            num_logical_experts=3,
            num_redundant_experts=1,
            max_slots_per_logical=4,
        )


def test_static_eplb_step_skips_runtime_rearrange(monkeypatch):
    class FakeGroup:
        def rank(self):
            return 0

    state = object.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            disable_online_rebalancing=True,
            expert_load_stats_path=None,
            log_balancedness_interval=1,
        )
    )
    state.model_states = {}
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 1
    state.expert_load_window_step = 0
    state.expert_load_window_size = 1
    state.is_async = False
    state.should_record_tensor = None
    state.rearrange = MagicMock()
    monkeypatch.setattr(
        eplb_state_module,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=FakeGroup()),
    )

    state.step()

    state.rearrange.assert_not_called()


def test_generate_static_mapping_from_stats(tmp_path, monkeypatch):
    script_path = (
        Path(__file__).parents[2] / "tools" / "eplb" / "generate_static_mapping.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_static_mapping", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stats_path = tmp_path / "stats.jsonl"
    output_path = tmp_path / "mapping.jsonl"
    stats_path.write_text(
        json.dumps(
            {
                "record_type": "eplb_load_stats",
                "version": 1,
                "step": 7,
                "num_ranks": 2,
                "expert_load": [[10, 1, 5, 2], [2, 8, 1, 3]],
                "p2l_map": [[0, 1, 2, 3], [0, 1, 2, 3]],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_static_mapping.py",
            "--stats-path",
            str(stats_path),
            "--output",
            str(output_path),
            "--num-redundant-experts",
            "2",
        ],
    )

    module.main()

    record = json.loads(output_path.read_text())
    assert record["record_type"] == "eplb_initial_mapping"
    assert record["version"] == 1
    assert record["num_redundant_experts"] == 2
    assert record["num_slots"] == 6
    assert set(record["initial_global_assignments"]) == {"0", "1"}
