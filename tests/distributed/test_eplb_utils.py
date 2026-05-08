# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_module
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
from vllm.distributed.eplb.stats_loader import (
    aggregate_logical_load,
    parse_stats_jsonl,
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


def test_static_eplb_step_skips_runtime_rearrange(monkeypatch):
    class FakeGroup:
        def rank(self):
            return 0

    state = object.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            enable_online=False,
            write_stats_path=None,
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
    state.total_steps = 0
    state.rearrange = MagicMock()
    monkeypatch.setattr(
        eplb_state_module,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=FakeGroup()),
    )

    state.step()

    state.rearrange.assert_not_called()


def _meta_record(num_layers: int = 2) -> dict:
    return {
        "record_type": "eplb_load_meta",
        "version": 1,
        "model": "test-model",
        "num_ranks": 2,
        "num_groups": 1,
        "num_nodes": 1,
        "num_layers": num_layers,
        "num_redundant_experts": 0,
    }


def test_parse_stats_jsonl_happy_path():
    text = "\n".join(
        [
            json.dumps(_meta_record()),
            json.dumps(
                {
                    "record_type": "eplb_load_stats",
                    "version": 1,
                    "step": 1,
                    "expert_load": [[10, 1, 5, 2], [2, 8, 1, 3]],
                    "p2l_map": [[0, 1, 2, 3], [0, 1, 2, 3]],
                }
            ),
            json.dumps(
                {
                    "record_type": "eplb_load_stats",
                    "version": 1,
                    "step": 2,
                    "expert_load": [[1, 1, 1, 1], [2, 2, 2, 2]],
                    "p2l_map": [[0, 1, 2, 3], [0, 1, 2, 3]],
                }
            ),
        ]
    )
    meta, stats = parse_stats_jsonl(text)
    assert meta["num_layers"] == 2
    assert len(stats) == 2
    assert stats[0]["step"] == 1
    assert stats[1]["step"] == 2


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "no eplb_load_meta record found"),
        (json.dumps(_meta_record()), "no eplb_load_stats records found"),
        (
            json.dumps(
                {
                    "record_type": "eplb_load_stats",
                    "version": 1,
                    "expert_load": [[1]],
                    "p2l_map": [[0]],
                }
            ),
            "before eplb_load_meta",
        ),
        (
            "\n".join([json.dumps(_meta_record()), json.dumps(_meta_record())]),
            "second eplb_load_meta",
        ),
        (
            json.dumps({"record_type": "garbage"}),
            "unexpected record_type",
        ),
    ],
)
def test_parse_stats_jsonl_rejects_corrupt_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_stats_jsonl(text)


def test_aggregate_logical_load_sums_across_records():
    records = [
        {
            "step": 1,
            # 2 layers, 4 physical slots, p2l identity (so logical == physical)
            "expert_load": [[10, 1, 5, 2], [2, 8, 1, 3]],
            "p2l_map": [[0, 1, 2, 3], [0, 1, 2, 3]],
        },
        {
            "step": 2,
            "expert_load": [[1, 1, 1, 1], [2, 2, 2, 2]],
            "p2l_map": [[0, 1, 2, 3], [0, 1, 2, 3]],
        },
    ]
    logical = aggregate_logical_load(records)
    assert torch.equal(
        logical, torch.tensor([[11.0, 2.0, 6.0, 3.0], [4.0, 10.0, 3.0, 5.0]])
    )


def test_aggregate_logical_load_collapses_replicas():
    # 2 physical slots both pointing at logical 0 -> their loads merge.
    records = [
        {
            "step": 1,
            "expert_load": [[3, 7]],
            "p2l_map": [[0, 0]],
        }
    ]
    logical = aggregate_logical_load(records)
    assert torch.equal(logical, torch.tensor([[10.0]]))
