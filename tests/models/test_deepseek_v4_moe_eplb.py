# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.moe import DeepseekV4MixtureOfExperts


class FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eplb_calls = []
        self.update_calls = 0

    def get_expert_weights(self):
        return [torch.ones(1)]

    def set_eplb_state(self, **kwargs) -> None:
        self.eplb_calls.append(kwargs)

    def update_expert_map(self) -> None:
        self.update_calls += 1


class FakeMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = FakeExperts()
        self.n_logical_experts = 8
        self.n_physical_experts = 10
        self.n_local_physical_experts = 5
        self.n_routed_experts = 8
        self.n_shared_experts = 1
        self.n_redundant_experts = 2


class UnsupportedExperts(nn.Module):
    pass


class FakeDecoder(nn.Module):
    def __init__(self, ffn: nn.Module) -> None:
        super().__init__()
        self.ffn = ffn


class FakeDeepseekV4(nn.Module, DeepseekV4MixtureOfExperts):
    decoder_layer_cls = FakeDecoder
    moe_layer_cls = FakeMoE

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"n_group": 2})()
        self.model = type("Model", (), {"layers": [FakeDecoder(FakeMoE())]})()
        self.set_moe_parameters()


def test_registers_layers_and_forwards_eplb_state():
    model = FakeDeepseekV4()

    assert model.num_moe_layers == 1
    assert model.num_logical_experts == 8
    assert model.num_physical_experts == 10

    expert_load_view = torch.ones(8)
    logical_to_physical_map = torch.arange(8)
    logical_replica_count = torch.ones(8, dtype=torch.int64)
    model.set_eplb_state(
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
    )

    assert len(model.expert_weights) == 1
    call = model.moe_layers[0].eplb_calls[0]
    assert call["moe_layer_idx"] == 0
    assert call["expert_load_view"] is expert_load_view
    assert call["logical_to_physical_map"] is logical_to_physical_map
    assert call["logical_replica_count"] is logical_replica_count


def test_refreshes_metadata_and_expert_map():
    model = FakeDeepseekV4()

    model.update_physical_experts_metadata(10, 5)

    moe = model.moe_mlp_layers[0]
    assert model.num_redundant_experts == 2
    assert moe.n_physical_experts == 10
    assert moe.n_local_physical_experts == 5
    assert moe.experts.update_calls == 1


def test_skips_layers_without_eplb_expert_methods():
    model = FakeDeepseekV4()
    model.model.layers = [FakeDecoder(FakeMoE())]
    model.model.layers[0].ffn.experts = UnsupportedExperts()
    model.set_moe_parameters()

    assert model.num_moe_layers == 0
    assert model.moe_layers == []
