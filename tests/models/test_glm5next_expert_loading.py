# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM5-next (GLM-5.3-Flash) and its MTP load routed experts through
`load_routed_experts_weight`: per-expert and transformers v5 fused names write
what the layer's own `RoutedExperts.load_weights` writes, with EPLB replicas,
and the ROCm fused shared expert lands in the slot after the routed experts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.kernels.moe import utils as moe_utils
from tests.kernels.moe.utils import (
    fake_moe_layer,
    make_routed_experts,
    routed_expert_checkpoint,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.models.glm5next.common import model as glm5next_model
from vllm.models.glm5next.common import mtp as glm5next_mtp

E, H, INTER = 8, 16, 12


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _config():
    return SimpleNamespace(
        n_routed_experts=E,
        n_shared_experts=1,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        mla_use_nope=False,
        qk_rope_head_dim=0,
        layer_types=None,
    )


class _Glm5Next(nn.Module):
    load_weights = glm5next_model.Glm5NextModel.load_weights
    # (checkpoint prefix, module prefix) of the MoE layer
    prefixes = ("layers.0", "layers.0")

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.config = _config()
        self.fused_shared_expert_layers: set[int] = set()
        self.layers = nn.ModuleList([layer])


class _Glm5NextMTP(nn.Module):
    load_weights = glm5next_mtp.Glm5NextMTP.load_weights
    _rewrite_spec_layer_name = glm5next_mtp.Glm5NextMTP._rewrite_spec_layer_name
    prefixes = ("model.layers.1", "model.layers.1.mtp_block")

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.config = _config()
        self.fused_shared_expert_layers: set[int] = set()
        self.model = nn.Module()
        self.model.mtp_start_layer_idx = 1
        self.model.num_mtp_layers = 1
        self.model.layers = nn.ModuleDict({"1": nn.Module()})
        self.model.layers["1"].mtp_block = layer


def _experts(model) -> nn.Module:
    return model.get_submodule(f"{model.prefixes[1]}.mlp.experts").routed_experts


def _use_routed_experts(layer: nn.Module, experts: nn.Module) -> nn.Module:
    for param in experts.parameters():
        param.data.fill_(-7.0)
    layer.mlp.experts.routed_experts = experts
    return layer


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


def _checkpoint(model, fused: bool, seed: int = 0):
    prefix = model.prefixes[0]
    experts = routed_expert_checkpoint(E, H, INTER, fused, seed=seed)
    return [(f"{prefix}.mlp.experts.{name}", tensor) for name, tensor in experts] + [
        (f"{prefix}.mlp.gate.weight", ROUTER)
    ]


def _assert_same(a: nn.Module, b: nn.Module):
    for (name, x), y in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), name


@pytest.mark.parametrize("model_cls", [_Glm5Next, _Glm5NextMTP])
@pytest.mark.parametrize(
    "tp,ep", [((1, 0), (1, 0)), ((2, 1), (1, 0)), ((1, 0), (2, 1))]
)
def test_routed_experts_load_through_the_moe_layer(model_cls, tp, ep):
    per_expert, fused, direct = (
        model_cls(fake_moe_layer(E, H, INTER, tp, ep)) for _ in range(3)
    )
    loaded = per_expert.load_weights(_checkpoint(per_expert, fused=False))
    # The main loader also takes (name, tensor, kwargs) tuples.
    fused_weights = _checkpoint(fused, fused=True)
    if model_cls is _Glm5Next:
        fused_weights = [(*weight, {}) for weight in fused_weights]
    fused.load_weights(fused_weights)
    list(_experts(direct).load_weights(routed_expert_checkpoint(E, H, INTER, False)))

    for model in (fused, direct):
        _assert_same(_experts(per_expert), _experts(model))
    assert torch.all(_experts(per_expert).w13_weight != -7.0)
    assert torch.all(_experts(per_expert).w2_weight != -7.0)
    module = model_cls.prefixes[1]
    assert torch.equal(per_expert.get_submodule(f"{module}.mlp.gate").weight, ROUTER)
    assert loaded == {
        f"{module}.mlp.experts.routed_experts.w13_weight",
        f"{module}.mlp.experts.routed_experts.w2_weight",
        f"{module}.mlp.gate.weight",
    }


@pytest.mark.parametrize("model_cls", [_Glm5Next, _Glm5NextMTP])
@pytest.mark.parametrize("fused", [False, True])
def test_eplb_replicas_are_loaded(model_cls, fused):
    """With 4 redundant experts on EP rank 1 of 2, every local physical slot
    (replicas of logical experts 0-3 included) gets its logical expert."""
    ep = (2, 1)
    model = model_cls(fake_moe_layer(E, H, INTER, ep=ep))
    experts = make_routed_experts(E, H, INTER, ep=ep, num_redundant_experts=4)
    _use_routed_experts(model.get_submodule(model.prefixes[1]), experts)
    model.load_weights(_checkpoint(model, fused=fused))

    reference = make_routed_experts(E, H, INTER)
    list(reference.load_weights(routed_expert_checkpoint(E, H, INTER, False)))
    # Physical experts 6-11 are local; 8-11 replicate logical experts 0-3.
    for local, logical in enumerate([6, 7, 0, 1, 2, 3]):
        assert torch.equal(experts.w13_weight[local], reference.w13_weight[logical])
        assert torch.equal(experts.w2_weight[local], reference.w2_weight[logical])


@pytest.mark.parametrize("model_cls", [_Glm5Next, _Glm5NextMTP])
def test_fused_shared_expert_slot(model_cls, monkeypatch):
    """ROCm AITER fused shared expert: `mlp.shared_experts.*` is renamed to
    expert `n_routed_experts` and loads into the slot after the routed ones."""
    monkeypatch.setattr(
        ExpertMapManager, "_init_aiter_shared_experts_topK_buffer", lambda self: None
    )
    monkeypatch.setattr(
        moe_utils,
        "ExpertMapManager",
        lambda *args, **kwargs: ExpertMapManager(
            *args, **kwargs, num_fused_shared_experts=1
        ),
    )
    monkeypatch.setattr(
        glm5next_model.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )
    model = model_cls(fake_moe_layer(E, H, INTER))
    model.fused_shared_expert_layers = {0 if model_cls is _Glm5Next else 1}
    experts = make_routed_experts(E, H, INTER)
    assert experts.w13_weight.shape[0] == E + 1
    _use_routed_experts(model.get_submodule(model.prefixes[1]), experts)

    shared = routed_expert_checkpoint(1, H, INTER, fused=False, seed=1)
    prefix = model.prefixes[0]
    model.load_weights(
        _checkpoint(model, fused=False)
        + [
            (f"{prefix}.mlp.shared_experts.{name.removeprefix('0.')}", tensor)
            for name, tensor in shared
        ]
    )

    # The references get a fused shared slot too (still patched).
    reference = make_routed_experts(E, H, INTER)
    list(reference.load_weights(routed_expert_checkpoint(E, H, INTER, False)))
    assert torch.equal(experts.w13_weight[:E], reference.w13_weight[:E])
    assert torch.equal(experts.w2_weight[:E], reference.w2_weight[:E])
    shared_reference = make_routed_experts(1, H, INTER)
    list(shared_reference.load_weights(shared))
    assert torch.equal(experts.w13_weight[E], shared_reference.w13_weight[0])
    assert torch.equal(experts.w2_weight[E], shared_reference.w2_weight[0])


def test_unmatched_expert_names_fall_through():
    model = _Glm5Next(fake_moe_layer(E, H, INTER))
    name = "layers.0.mlp.experts.7.w1.weight"
    with pytest.raises(KeyError):
        model.load_weights([(name, torch.zeros(INTER, H))])
