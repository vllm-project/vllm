# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3MegaMoEExperts,
    KimiLinearForCausalLM,
    KimiMoE,
)


class FakeRunner:
    def __init__(self) -> None:
        self.weight = torch.empty(2, 1)
        self.state_calls: list[dict[str, object]] = []
        self.update_calls = 0

    def get_expert_weights(self) -> list[torch.Tensor]:
        return [self.weight]

    def set_eplb_state(self, **kwargs) -> None:
        self.state_calls.append(kwargs)

    def update_expert_map(self) -> None:
        self.update_calls += 1


class FakeMoE(KimiMoE):
    def __init__(self, logical: int = 8, redundant: int = 2) -> None:
        self.n_logical_experts = logical
        self.n_physical_experts = logical + redundant
        self.n_local_physical_experts = (logical + redundant) // 2
        self.n_routed_experts = logical
        self.n_shared_experts = 1
        self.n_redundant_experts = redundant
        self.experts = FakeRunner()


class FakeLayer:
    def __init__(self, mlp) -> None:
        self.mlp = mlp


class FakeKimiModel(KimiLinearForCausalLM):
    def __init__(self, layers) -> None:
        self.config = SimpleNamespace(num_expert_group=1)
        self.model = SimpleNamespace(layers=layers)


def test_kimi_eplb_discovers_only_local_moe_layers_in_order():
    first = FakeMoE()
    second = FakeMoE()
    model = FakeKimiModel(
        [
            FakeLayer(object()),
            FakeLayer(first),
            PPMissingLayer(),
            FakeLayer(second),
        ]
    )

    model.set_moe_parameters()

    assert model.moe_mlp_layers == [first, second]
    assert model.moe_layers == [first.experts, second.experts]
    assert model.num_moe_layers == 2
    assert model.num_logical_experts == 8
    assert model.num_physical_experts == 10
    assert model.num_local_physical_experts == 5
    assert model.num_routed_experts == 8
    assert model.num_shared_experts == 1
    assert model.num_redundant_experts == 2


def test_kimi_eplb_stage_without_moe_exposes_zero_metadata():
    model = FakeKimiModel([FakeLayer(object()), PPMissingLayer()])

    model.set_moe_parameters()

    assert model.num_moe_layers == 0
    assert model.moe_layers == []
    assert model.num_expert_groups == 0
    assert model.num_logical_experts == 0
    assert model.num_physical_experts == 0
    assert model.num_local_physical_experts == 0
    assert model.num_routed_experts == 0
    assert model.num_shared_experts == 0
    assert model.num_redundant_experts == 0
    assert not is_mixture_of_experts(model)


def test_kimi_eplb_model_satisfies_protocol_and_propagates_layer_state():
    moe = FakeMoE()
    model = FakeKimiModel([FakeLayer(moe)])
    model.set_moe_parameters()
    expert_load = torch.zeros(1, 10, dtype=torch.int32)
    logical_to_physical = torch.zeros(1, 8, 2, dtype=torch.int32)
    replica_count = torch.ones(1, 8, dtype=torch.int32)

    assert is_mixture_of_experts(model)
    model.set_eplb_state(expert_load, logical_to_physical, replica_count)

    assert model.expert_weights == [[moe.experts.weight]]
    assert moe.experts.state_calls == [
        {
            "moe_layer_idx": 0,
            "expert_load_view": expert_load,
            "logical_to_physical_map": logical_to_physical,
            "logical_replica_count": replica_count,
        }
    ]


def test_kimi_eplb_updates_all_layer_metadata_and_dispatch_maps():
    first = FakeMoE()
    second = FakeMoE()
    model = FakeKimiModel([FakeLayer(first), FakeLayer(second)])
    model.set_moe_parameters()

    model.update_physical_experts_metadata(
        num_physical_experts=12,
        num_local_physical_experts=5,
    )

    assert model.num_physical_experts == 12
    assert model.num_redundant_experts == 4
    for moe in (first, second):
        assert moe.n_physical_experts == 12
        assert moe.n_local_physical_experts == 5
        assert moe.n_redundant_experts == 4
        assert moe.experts.update_calls == 1


def test_kimi_mega_moe_initial_mapping_duplicates_logical_experts():
    experts = object.__new__(KimiK3MegaMoEExperts)
    experts.num_logical_experts = 4
    experts.experts_start_idx = 3
    experts.experts_end_idx = 8

    assert experts._map_global_expert_id(0) == [1]
    assert experts._map_global_expert_id(3) == [0, 4]


def test_kimi_mega_moe_updates_elastic_ep_metadata(monkeypatch):
    experts = object.__new__(KimiK3MegaMoEExperts)
    experts.num_local_experts = 5
    experts.num_experts = 10
    experts.experts_start_idx = 5
    experts.experts_end_idx = 10
    monkeypatch.setattr(
        kimi_model,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=3, rank_in_group=2),
    )

    experts.update_expert_map()

    assert experts.num_experts == 15
    assert experts.experts_start_idx == 10
    assert experts.experts_end_idx == 15


def test_kimi_eplb_expert_counts_include_redundancy():
    assert kimi_model._get_kimi_eplb_expert_counts(896, 32, 8) == (928, 116)


def test_kimi_eplb_expert_counts_reject_uneven_ep_distribution():
    with pytest.raises(
        ValueError,
        match="num_physical_experts=898 must be divisible by EP size 8",
    ):
        kimi_model._get_kimi_eplb_expert_counts(896, 2, 8)


def test_kimi_weight_mapping_initializes_redundant_fused_moe_slots(
    monkeypatch,
):
    captured = {}

    def fake_mapping(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        kimi_model,
        "fused_moe_make_expert_params_mapping",
        fake_mapping,
    )
    model = object.__new__(kimi_model.KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        is_moe=True,
        linear_attn_config=None,
        num_experts=896,
        q_lora_rank=None,
    )
    model.num_redundant_experts = 32

    assert model.load_weights([]) == set()
    assert captured["num_experts"] == 896
    assert captured["num_redundant_experts"] == 32
