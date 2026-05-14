# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.utils.moe_pruned_experts import (
    build_pruned_expert_map,
    load_pruned_experts_profile,
)


def test_load_pruned_experts_profile_groups_pruned_experts_by_layer(tmp_path):
    profile_path = tmp_path / "pruned_experts.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "pruned_experts": [[0, 2], [0, 5], [3, 7]],
            }
        )
    )

    profile = load_pruned_experts_profile(str(profile_path))

    assert profile.pruned_experts_by_layer == {0: frozenset({2, 5}), 3: frozenset({7})}


def test_build_pruned_expert_map_compacts_unpruned_experts():
    expert_map, num_local_experts = build_pruned_expert_map(
        num_experts=8,
        pruned_experts=frozenset({2, 5}),
    )

    assert num_local_experts == 6
    assert expert_map.dtype == torch.int32
    assert expert_map.tolist() == [0, 1, -1, 2, 3, -1, 4, 5]


def test_build_pruned_expert_map_composes_with_existing_expert_map():
    base_expert_map = torch.tensor([0, 1, -1, -1, 2, 3], dtype=torch.int32)

    expert_map, num_local_experts = build_pruned_expert_map(
        num_experts=6,
        pruned_experts=frozenset({1, 4}),
        base_expert_map=base_expert_map,
    )

    assert num_local_experts == 2
    assert expert_map.tolist() == [0, -1, -1, -1, -1, 1]


def test_load_pruned_experts_profile_rejects_invalid_expert_entry(tmp_path):
    profile_path = tmp_path / "pruned_experts.json"
    profile_path.write_text(json.dumps({"version": 1, "pruned_experts": [[0]]}))

    with pytest.raises(ValueError, match="layer, expert"):
        load_pruned_experts_profile(str(profile_path))
