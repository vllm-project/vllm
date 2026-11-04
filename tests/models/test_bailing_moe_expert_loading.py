# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ling / Bailing (V2.5 linear, V3) and their MTP load routed experts through
`load_routed_experts_weight`: per-expert and transformers v5 fused names
write what the layer's own `RoutedExperts.load_weights` writes, also behind a
LoRA wrapper and with EPLB replicas; gate biases checkpointed under
`.experts.` still reach the gate."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.kernels.moe.utils import (
    fake_moe_layer,
    make_routed_experts,
    routed_expert_checkpoint,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.models.bailing_moe_linear import BailingMoeV25Model
from vllm.model_executor.models.bailing_moe_mtp import BailingMoeV25MTPModel
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from vllm.model_executor.models.bailing_moe_v3_mtp import BailingMoeV3MTPModel

E, H, INTER = 8, 16, 12


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _moe_layer(tp, ep, lora=False, num_redundant_experts=0) -> nn.Module:
    layer = fake_moe_layer(E, H, INTER, tp, ep, lora)
    layer.mlp.gate.expert_bias = nn.Parameter(torch.zeros(E), requires_grad=False)
    if num_redundant_experts:
        experts = make_routed_experts(
            E, H, INTER, tp, ep, num_redundant_experts=num_redundant_experts
        )
        for param in experts.parameters():
            param.data.fill_(-7.0)
        layer.mlp.experts.routed_experts = experts
    return layer


def _config(**kwargs):
    return SimpleNamespace(
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        num_experts=E,
        layer_group_size=1,
        **kwargs,
    )


class _V3(nn.Module):
    load_weights = BailingMoeV3ForCausalLM.load_weights
    hf_to_vllm_mapper = BailingMoeV3ForCausalLM.hf_to_vllm_mapper
    # (checkpoint prefix, module prefix) of the MoE layer
    prefixes = ("model.layers.0.mlp", "model.layers.0.mlp")
    quant_config = None

    def __init__(self, tp, ep, lora=False, num_redundant_experts=0):
        super().__init__()
        self.config = _config()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_moe_layer(tp, ep, lora, num_redundant_experts)]
        )


class _V3MTP(nn.Module):
    load_weights = BailingMoeV3MTPModel.load_weights
    hf_to_vllm_mapper = BailingMoeV3MTPModel.hf_to_vllm_mapper
    prefixes = ("model.layers.1.mlp", "model.layers.1.mlp")
    quant_config = None

    def __init__(self, tp, ep, lora=False, num_redundant_experts=0):
        super().__init__()
        self.config = _config()
        self.model = nn.Module()
        self.model.mtp_start_layer_idx = 1
        self.model.num_mtp_layers = 1
        self.model.layers = nn.ModuleDict(
            {"1": _moe_layer(tp, ep, lora, num_redundant_experts)}
        )


class _V25(nn.Module):
    """`BailingMoeV25Model`, which `AutoWeightsLoader` hands names without
    `model.`; it strips the prefix itself too."""

    load_weights = BailingMoeV25Model.load_weights
    prefixes = ("model.layers.0.mlp", "layers.0.mlp")

    def __init__(self, tp, ep, lora=False, num_redundant_experts=0):
        super().__init__()
        self.config = _config()
        self.layers = nn.ModuleList([_moe_layer(tp, ep, lora, num_redundant_experts)])


class _V25MTP(_V3MTP):
    load_weights = BailingMoeV25MTPModel.load_weights


MODELS = [_V3, _V3MTP, _V25, _V25MTP]


def _experts(model) -> RoutedExperts:
    runner = model.get_submodule(f"{model.prefixes[1]}.experts")
    return getattr(runner, "base_layer", runner).routed_experts


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))
BIAS = torch.randn(E, generator=torch.Generator().manual_seed(2))


def _checkpoint(model, fused: bool, suffix: str = "weight"):
    prefix = model.prefixes[0]
    experts = routed_expert_checkpoint(E, H, INTER, fused, suffix)
    return [(f"{prefix}.experts.{name}", tensor) for name, tensor in experts] + [
        (f"{prefix}.gate.weight", ROUTER)
    ]


def _expected_names(model, base_layer: str = "") -> set[str]:
    module = model.prefixes[1]
    return {
        f"{module}.experts.{base_layer}routed_experts.w13_weight",
        f"{module}.experts.{base_layer}routed_experts.w2_weight",
        f"{module}.gate.weight",
    }


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize(
    "tp,ep,redundant",
    [
        ((1, 0), (1, 0), 0),
        ((2, 1), (1, 0), 0),
        ((1, 0), (2, 1), 0),
        ((1, 0), (2, 0), 2),
    ],
)
def test_routed_experts_load_through_the_moe_layer(model_cls, tp, ep, redundant):
    per_expert, fused, direct = (model_cls(tp, ep, False, redundant) for _ in range(3))
    loaded = per_expert.load_weights(_checkpoint(per_expert, fused=False))
    fused.load_weights(_checkpoint(fused, fused=True))
    list(_experts(direct).load_weights(routed_expert_checkpoint(E, H, INTER, False)))

    for model in (fused, direct):
        for (name, a), b in zip(
            _experts(per_expert).named_parameters(), _experts(model).parameters()
        ):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    # EPLB replica slots are filled too.
    assert torch.all(_experts(per_expert).w13_weight != -7.0)
    assert torch.all(_experts(per_expert).w2_weight != -7.0)
    module = model_cls.prefixes[1]
    assert torch.equal(per_expert.get_submodule(f"{module}.gate").weight, ROUTER)
    assert loaded == _expected_names(per_expert)


@pytest.mark.parametrize("model_cls", MODELS)
def test_lora_wrapped_moe_layer_is_unwrapped(model_cls):
    model = model_cls((1, 0), (1, 0), lora=True)
    loaded = model.load_weights(
        _checkpoint(model, fused=False, suffix="base_layer.weight")
    )
    assert loaded == _expected_names(model, base_layer="base_layer.")
    assert loaded <= dict(model.named_parameters()).keys()
    reference = make_routed_experts(E, H, INTER)
    list(reference.load_weights(routed_expert_checkpoint(E, H, INTER, False)))
    assert torch.equal(_experts(model).w13_weight, reference.w13_weight)
    assert torch.equal(_experts(model).w2_weight, reference.w2_weight)


@pytest.mark.parametrize("bias_name", ["e_score_correction_bias", "expert_bias"])
def test_gate_bias_under_experts_reaches_the_gate(bias_name):
    """Bailing V2.5 checkpoints may store the router bias as
    `mlp.experts.<bias_name>`; it is not an expert weight and the model's
    own redirect loads it into `mlp.gate.expert_bias`."""
    model = _V25((1, 0), (1, 0))
    loaded = model.load_weights(
        _checkpoint(model, fused=False)
        + [(f"model.layers.0.mlp.experts.{bias_name}", BIAS)]
    )
    assert loaded == _expected_names(model) | {"layers.0.mlp.gate.expert_bias"}
    assert torch.equal(model.layers[0].mlp.gate.expert_bias, BIAS)


@pytest.mark.parametrize("model_cls", [_V3MTP, _V25MTP])
def test_mtp_layer_with_only_remote_experts_counts_as_loaded(model_cls):
    """Under EP an expert tensor of another rank matches the layer but loads
    nothing; it still marks the MTP layer as present, as the old loop did."""
    model = model_cls((1, 0), (2, 1))
    name = f"{model.prefixes[0]}.experts.0.gate_proj.weight"
    assert model.load_weights([(name, torch.zeros(INTER, H))]) == set()
    with pytest.raises(ValueError, match="missing"):
        model.load_weights([])


@pytest.mark.parametrize("model_cls", MODELS)
def test_unmatched_expert_names_are_skipped(model_cls):
    """Names under `.experts.` that match no mapping entry are skipped, as
    the base loop skipped them."""
    model = model_cls((1, 0), (1, 0))
    name = f"{model.prefixes[0]}.experts.0.w1.weight"
    checkpoint = _checkpoint(model, fused=False) + [(name, torch.zeros(INTER, H))]
    assert model.load_weights(checkpoint) == _expected_names(model)
