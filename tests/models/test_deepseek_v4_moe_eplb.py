# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.models.deepseek_v4.common.moe import DeepseekV4MixtureOfExperts


class FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.update_expert_map_calls = 0

    def get_expert_weights(self):
        return [torch.empty(1)]

    def set_eplb_state(self, *args, **kwargs) -> None:
        pass

    def update_expert_map(self) -> None:
        self.update_expert_map_calls += 1


class UnsupportedExperts(nn.Module):
    pass


class FakeMoE(nn.Module):
    def __init__(self, experts: nn.Module | None = None) -> None:
        super().__init__()
        self.n_logical_experts = 8
        self.n_physical_experts = 10
        self.n_local_physical_experts = 5
        self.n_routed_experts = 8
        self.n_shared_experts = 1
        self.n_redundant_experts = 2
        self.experts = experts if experts is not None else FakeExperts()


class FakeDense(nn.Module):
    pass


class FakeDecoderLayer(nn.Module):
    def __init__(self, ffn: nn.Module) -> None:
        super().__init__()
        self.ffn = ffn


class FakeModel(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class FakeDeepseekV4ForCausalLM(nn.Module, DeepseekV4MixtureOfExperts):
    decoder_layer_cls = FakeDecoderLayer
    moe_layer_cls = FakeMoE

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=len(layers), n_group=4)
        self.model = FakeModel(layers)
        self.set_moe_parameters()


def test_deepseek_v4_moe_mixin_registers_eplb_capable_layers():
    first_moe = FakeMoE()
    second_moe = FakeMoE()
    model = FakeDeepseekV4ForCausalLM(
        [
            FakeDecoderLayer(first_moe),
            FakeDecoderLayer(FakeDense()),
            FakeDecoderLayer(second_moe),
        ]
    )

    assert is_mixture_of_experts(model)
    assert model.num_moe_layers == 2
    assert model.num_expert_groups == 4
    assert model.num_logical_experts == 8
    assert model.num_physical_experts == 10
    assert model.num_local_physical_experts == 5
    assert model.num_routed_experts == 8
    assert model.num_shared_experts == 1
    assert model.num_redundant_experts == 2
    assert model.moe_layers == [first_moe.experts, second_moe.experts]

    model.update_physical_experts_metadata(
        num_physical_experts=12,
        num_local_physical_experts=5,
    )

    assert model.num_physical_experts == 12
    assert model.num_redundant_experts == 4
    assert first_moe.n_physical_experts == 12
    assert second_moe.n_redundant_experts == 4
    assert first_moe.experts.update_expert_map_calls == 1
    assert second_moe.experts.update_expert_map_calls == 1


def test_deepseek_v4_moe_mixin_skips_non_eplb_expert_layers():
    model = FakeDeepseekV4ForCausalLM([FakeDecoderLayer(FakeMoE(UnsupportedExperts()))])

    assert not is_mixture_of_experts(model)
    assert model.num_moe_layers == 0
    assert model.moe_layers == []


def test_deepseek_v4_platform_models_share_moe_mixin():
    repo_root = Path(__file__).parents[2]
    for model_path in (
        repo_root / "vllm/models/deepseek_v4/nvidia/model.py",
        repo_root / "vllm/models/deepseek_v4/amd/model.py",
        repo_root / "vllm/models/deepseek_v4/xpu/model.py",
    ):
        module = ast.parse(model_path.read_text())
        cls = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4ForCausalLM"
        )
        assert any(
            isinstance(base, ast.Name) and base.id == "DeepseekV4MixtureOfExperts"
            for base in cls.bases
        ), model_path
