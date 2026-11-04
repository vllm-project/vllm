# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY-V4 (NVIDIA and ROCm) and its MTP load routed experts through
`load_routed_experts_weight`: per-expert and transformers v5 fused names write
what the layer's own `RoutedExperts.load_weights` writes, with EPLB replicas,
also behind a LoRA wrapper."""

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
from vllm.models.hy_v4.amd.model import HYV4Model as HYV4ModelROCm
from vllm.models.hy_v4.nvidia.model import HYV4Model
from vllm.models.hy_v4.nvidia.mtp import HYV4MTP

E, H, INTER = 8, 16, 12


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


class _HYV4(nn.Module):
    load_weights = HYV4Model.load_weights
    # (checkpoint prefix, module prefix) of the MoE layer
    prefixes = ("layers.0", "layers.0")

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, num_experts=E)
        self.layers = nn.ModuleList([layer])


class _HYV4ROCm(_HYV4):
    load_weights = HYV4ModelROCm.load_weights


class _HYV4MTP(nn.Module):
    load_weights = HYV4MTP.load_weights
    _rewrite_spec_layer_name = HYV4MTP._rewrite_spec_layer_name
    prefixes = ("model.mtp_layers.0", "model.layers.1.mtp_block")

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=1, num_nextn_predict_layers=1, n_routed_experts=E
        )
        self.quant_config = None
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({"1": nn.Module()})
        self.model.layers["1"].mtp_block = layer


MODELS = [_HYV4, _HYV4ROCm, _HYV4MTP]


def _experts(model) -> nn.Module:
    runner = model.get_submodule(f"{model.prefixes[1]}.mlp.experts")
    return getattr(runner, "base_layer", runner).routed_experts


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


def _checkpoint(model, fused: bool, suffix: str = "weight"):
    prefix = model.prefixes[0]
    experts = routed_expert_checkpoint(E, H, INTER, fused, suffix)
    return [(f"{prefix}.mlp.experts.{name}", tensor) for name, tensor in experts] + [
        (f"{prefix}.mlp.gate.weight", ROUTER)
    ]


def _reference():
    reference = make_routed_experts(E, H, INTER)
    list(reference.load_weights(routed_expert_checkpoint(E, H, INTER, False)))
    return reference


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize(
    "tp,ep", [((1, 0), (1, 0)), ((2, 1), (1, 0)), ((1, 0), (2, 1))]
)
def test_routed_experts_load_through_the_moe_layer(model_cls, tp, ep):
    per_expert, fused, direct = (
        model_cls(fake_moe_layer(E, H, INTER, tp, ep)) for _ in range(3)
    )
    loaded = per_expert.load_weights(_checkpoint(per_expert, fused=False))
    fused.load_weights(_checkpoint(fused, fused=True))
    list(_experts(direct).load_weights(routed_expert_checkpoint(E, H, INTER, False)))

    for model in (fused, direct):
        for (name, a), b in zip(
            _experts(per_expert).named_parameters(), _experts(model).parameters()
        ):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    assert torch.all(_experts(per_expert).w13_weight != -7.0)
    assert torch.all(_experts(per_expert).w2_weight != -7.0)
    module = model_cls.prefixes[1]
    assert torch.equal(per_expert.get_submodule(f"{module}.mlp.gate").weight, ROUTER)
    assert loaded == {
        f"{module}.mlp.experts.routed_experts.w13_weight",
        f"{module}.mlp.experts.routed_experts.w2_weight",
        f"{module}.mlp.gate.weight",
    }


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("fused", [False, True])
def test_eplb_replicas_are_loaded(model_cls, fused):
    """With 4 redundant experts on EP rank 1 of 2, every local physical slot
    (replicas of logical experts 0-3 included) gets its logical expert."""
    ep = (2, 1)
    model = model_cls(fake_moe_layer(E, H, INTER, ep=ep))
    experts = make_routed_experts(E, H, INTER, ep=ep, num_redundant_experts=4)
    for param in experts.parameters():
        param.data.fill_(-7.0)
    model.get_submodule(f"{model.prefixes[1]}.mlp.experts").routed_experts = experts
    model.load_weights(_checkpoint(model, fused=fused))

    reference = _reference()
    # Physical experts 6-11 are local; 8-11 replicate logical experts 0-3.
    for local, logical in enumerate([6, 7, 0, 1, 2, 3]):
        assert torch.equal(experts.w13_weight[local], reference.w13_weight[logical])
        assert torch.equal(experts.w2_weight[local], reference.w2_weight[logical])


@pytest.mark.parametrize("model_cls", [_HYV4, _HYV4ROCm])
@pytest.mark.parametrize("fused", [False, True])
def test_lora_wrapped_moe_layer_is_unwrapped(model_cls, fused):
    model = model_cls(fake_moe_layer(E, H, INTER, lora=True))
    loaded = model.load_weights(
        _checkpoint(model, fused=fused, suffix="base_layer.weight")
    )
    prefix = "layers.0.mlp.experts.base_layer.routed_experts"
    assert loaded == {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
        "layers.0.mlp.gate.weight",
    }
    assert loaded <= dict(model.named_parameters()).keys()
    reference = _reference()
    assert torch.equal(_experts(model).w13_weight, reference.w13_weight)
    assert torch.equal(_experts(model).w2_weight, reference.w2_weight)


def test_unquantized_mtp_skips_expert_scales():
    model = _HYV4MTP(fake_moe_layer(E, H, INTER))
    prefix = model.prefixes[0]
    loaded = model.load_weights(
        _checkpoint(model, fused=False)
        + [
            (f"{prefix}.mlp.experts.0.gate_proj.weight_scale_inv", torch.ones(1, 1)),
            (f"{prefix}.mlp.experts.gate_up_proj_scale_inv", torch.ones(E, 2, 1)),
        ]
    )
    assert loaded == dict(model.named_parameters()).keys()
    reference = _reference()
    assert torch.equal(_experts(model).w13_weight, reference.w13_weight)


@pytest.mark.parametrize("model_cls", [_HYV4, _HYV4ROCm])
def test_unmatched_expert_names_fall_through(model_cls):
    """A name under `experts` that matches no mapping entry is an unknown
    checkpoint weight, skipped with a warning as before."""
    model = model_cls(fake_moe_layer(E, H, INTER))
    name = "layers.0.mlp.experts.7.w1.weight"
    assert model.load_weights([(name, torch.zeros(INTER, H))]) == set()
