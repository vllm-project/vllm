# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY-V3 and its MTP load routed experts through `load_routed_experts_weight`:
per-expert and transformers v5 fused names write what the layer's own
`RoutedExperts.load_weights` writes, also behind a LoRA wrapper."""

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
from vllm.model_executor.layers.fused_moe import load_routed_experts_weight
from vllm.model_executor.models.hy_v3 import HYV3Model
from vllm.model_executor.models.hy_v3_mtp import HYV3MTP

E, H, INTER = 8, 16, 12


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


class _HYV3(nn.Module):
    load_weights = HYV3Model.load_weights
    # (checkpoint prefix, module prefix) of the MoE layer
    prefixes = ("layers.0", "layers.0")

    def __init__(self, tp, ep, lora=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, num_experts=E)
        self.layers = nn.ModuleList([fake_moe_layer(E, H, INTER, tp, ep, lora)])


class _HYV3MTP(nn.Module):
    load_weights = HYV3MTP.load_weights
    _rewrite_spec_layer_name = HYV3MTP._rewrite_spec_layer_name
    _split_qkv_weight = HYV3MTP._split_qkv_weight
    prefixes = ("model.layers.1", "model.layers.1.mtp_block")

    def __init__(self, tp, ep, lora=False):
        super().__init__()
        self.config = SimpleNamespace(
            tie_word_embeddings=False,
            num_experts=E,
            num_hidden_layers=1,
            num_nextn_predict_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        self.quant_config = None
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({"1": nn.Module()})
        self.model.layers["1"].mtp_block = fake_moe_layer(E, H, INTER, tp, ep, lora)


def _experts(model) -> nn.Module:
    runner = model.get_submodule(f"{model.prefixes[1]}.mlp.experts")
    return getattr(runner, "base_layer", runner).routed_experts


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


def _checkpoint(model, fused: bool, suffix: str = "weight"):
    prefix = model.prefixes[0]
    experts = routed_expert_checkpoint(E, H, INTER, fused, suffix)
    return [(f"{prefix}.mlp.experts.{name}", tensor) for name, tensor in experts] + [
        (f"{prefix}.mlp.router.gate.weight", ROUTER)
    ]


@pytest.mark.parametrize("model_cls", [_HYV3, _HYV3MTP])
@pytest.mark.parametrize(
    "tp,ep", [((1, 0), (1, 0)), ((2, 1), (1, 0)), ((1, 0), (2, 1))]
)
def test_routed_experts_load_through_the_moe_layer(model_cls, tp, ep):
    per_expert, fused, direct = (model_cls(tp, ep) for _ in range(3))
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
    if model_cls is _HYV3:
        assert loaded == {
            f"{module}.mlp.experts.routed_experts.w13_weight",
            f"{module}.mlp.experts.routed_experts.w2_weight",
            f"{module}.mlp.gate.weight",
        }


def test_lora_wrapped_moe_layer_is_unwrapped():
    model = _HYV3((1, 0), (1, 0), lora=True)
    loaded = model.load_weights(
        _checkpoint(model, fused=False, suffix="base_layer.weight")
    )
    prefix = "layers.0.mlp.experts.base_layer.routed_experts"
    assert loaded == {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
        "layers.0.mlp.gate.weight",
    }
    assert loaded <= dict(model.named_parameters()).keys()
    reference = make_routed_experts(E, H, INTER)
    list(reference.load_weights(routed_expert_checkpoint(E, H, INTER, False)))
    assert torch.equal(_experts(model).w13_weight, reference.w13_weight)
    assert torch.equal(_experts(model).w2_weight, reference.w2_weight)


def test_unmatched_expert_names_fall_through():
    model = _HYV3((1, 0), (2, 1))
    # Expert 0 is on the other EP rank: handled, nothing loaded.
    name = "layers.0.mlp.experts.0.gate_proj.weight"
    assert load_routed_experts_weight(model, name, torch.zeros(INTER, H)) == set()
    # No mapping entry: the caller falls through to its own lookup.
    name = "layers.0.mlp.experts.7.w1.weight"
    assert load_routed_experts_weight(model, name, torch.zeros(INTER, H)) is None
    with pytest.raises(KeyError):
        model.load_weights([(name, torch.zeros(INTER, H))])
