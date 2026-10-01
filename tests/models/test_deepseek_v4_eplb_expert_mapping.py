# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FusedMoE expert mapping of DeepSeek V4 / V4.1 must cover the EPLB
redundant physical expert slots, or those slots are never loaded."""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV4Model as DeepseekV41Model,
)


def _fake_model(n_routed_experts: int, n_redundant_experts: int):
    ffn = SimpleNamespace(use_mega_moe=False, n_redundant_experts=n_redundant_experts)
    return SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=n_routed_experts),
        layers=[SimpleNamespace(ffn=ffn)],
        start_layer=0,
        end_layer=1,
        named_parameters=lambda: iter(()),
    )


@pytest.mark.parametrize(
    "model_cls", [DeepseekV4Model, DeepseekV41Model], ids=["v4", "v41"]
)
@pytest.mark.parametrize("n_redundant_experts", [0, 3])
def test_expert_mapping_covers_redundant_slots(model_cls, n_redundant_experts):
    n_routed_experts = 4
    mapping = model_cls.get_expert_mapping(
        _fake_model(n_routed_experts, n_redundant_experts)
    )

    physical_ids = {expert_id for _, _, expert_id, _ in mapping}
    assert physical_ids == set(range(n_routed_experts + n_redundant_experts))

    # Each redundant slot loads a real checkpoint expert (the initial
    # placement replicates logical expert i % n_routed_experts).
    for _, weight_name, expert_id, shard_id in mapping:
        logical_id = expert_id % n_routed_experts
        assert weight_name == f"experts.{logical_id}.{shard_id}."
