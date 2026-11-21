# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert gate/up tensors renamed by stacked mappers must not be fused (CPU)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts import (
    _unstack_per_expert_gate_up,
)
from vllm.model_executor.models.utils import WeightsMapper


def _tagged(shard_id):
    t = torch.empty(4, 8)
    if shard_id is not None:
        t.shard_id = shard_id
    return t


@pytest.mark.parametrize(
    "name,shard_id,expected",
    [
        ("3.gate_up_proj.weight", 0, "3.gate_proj.weight"),
        ("3.gate_up_proj.weight", 1, "3.up_proj.weight"),
        ("experts.17.gate_up_proj.qweight", 1, "experts.17.up_proj.qweight"),
        # Genuine per-expert fused tensors carry no shard_id: untouched.
        ("3.gate_up_proj.weight", None, "3.gate_up_proj.weight"),
        # Fused all-expert tensors have no expert id: untouched.
        ("gate_up_proj", 0, "gate_up_proj"),
        ("experts.gate_up_proj.weight", 1, "experts.gate_up_proj.weight"),
        ("3.down_proj.weight", 0, "3.down_proj.weight"),
    ],
)
def test_unstack_per_expert_gate_up(name, shard_id, expected):
    assert _unstack_per_expert_gate_up(name, _tagged(shard_id)) == expected


def test_llama_style_mapper_round_trip():
    """LlamaModel-style stacking tags per-expert halves; the helper undoes it."""
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
    weights = [
        ("layers.0.mlp.experts.5.gate_proj.weight", torch.zeros(4, 8)),
        ("layers.0.mlp.experts.5.up_proj.weight", torch.ones(4, 8)),
    ]
    restored = [
        _unstack_per_expert_gate_up(name.split("experts.", 1)[1], t)
        for name, t in mapper.apply(weights)
    ]
    assert restored == ["5.gate_proj.weight", "5.up_proj.weight"]
