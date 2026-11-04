# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V3.2 / GLM-5.x DSA (NVIDIA and ROCm) and their MTP load routed
experts through `load_routed_experts_weight`: per-expert and transformers v5
fused names write what the layer's own `RoutedExperts.load_weights` writes,
with EPLB replicas, and on ROCm the widened fused shared expert is split into
the slots after the routed experts."""

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
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.models.deepseek_v32.amd import model as amd_model
from vllm.models.deepseek_v32.amd import mtp as amd_mtp
from vllm.models.deepseek_v32.nvidia import model as nvidia_model
from vllm.models.deepseek_v32.nvidia import mtp as nvidia_mtp

E, H, INTER = 8, 16, 12


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _config(n_shared_experts: int = 1):
    return SimpleNamespace(
        n_routed_experts=E,
        n_shared_experts=n_shared_experts,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )


class _Model(nn.Module):
    # (checkpoint prefix, module prefix) of the MoE layer
    prefixes = ("layers.0", "layers.0")

    def __init__(self, layer: nn.Module, n_shared_experts: int = 1):
        super().__init__()
        self.config = _config(n_shared_experts)
        self.fused_shared_expert_layers: set[int] = set()
        self.layers = nn.ModuleList([layer])


class _NvidiaModel(_Model):
    load_weights = nvidia_model.DeepseekV32Model.load_weights


class _AmdModel(_Model):
    load_weights = amd_model.DeepseekV32Model.load_weights


class _MTP(nn.Module):
    prefixes = ("model.layers.1", "model.layers.1.mtp_block")

    def __init__(self, layer: nn.Module, n_shared_experts: int = 1):
        super().__init__()
        self.config = _config(n_shared_experts)
        self.fused_shared_expert_layers: set[int] = set()
        self.model = nn.Module()
        self.model.mtp_start_layer_idx = 1
        self.model.num_mtp_layers = 1
        self.model.layers = nn.ModuleDict({"1": nn.Module()})
        self.model.layers["1"].mtp_block = layer


class _NvidiaMTP(_MTP):
    load_weights = nvidia_mtp.DeepseekV32MTP.load_weights
    _rewrite_spec_layer_name = nvidia_mtp.DeepseekV32MTP._rewrite_spec_layer_name
    is_unused_checkpoint_weight = nvidia_mtp.DeepseekV32MTP.is_unused_checkpoint_weight


class _AmdMTP(_MTP):
    load_weights = amd_mtp.DeepseekV32MTP.load_weights
    _rewrite_spec_layer_name = amd_mtp.DeepseekV32MTP._rewrite_spec_layer_name
    is_unused_checkpoint_weight = amd_mtp.DeepseekV32MTP.is_unused_checkpoint_weight


MODELS = [_NvidiaModel, _AmdModel, _NvidiaMTP, _AmdMTP]


def _experts(model) -> nn.Module:
    return model.get_submodule(f"{model.prefixes[1]}.mlp.experts").routed_experts


def _use_routed_experts(model, experts: nn.Module) -> nn.Module:
    for param in experts.parameters():
        param.data.fill_(-7.0)
    model.get_submodule(f"{model.prefixes[1]}.mlp.experts").routed_experts = experts
    return experts


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


def _checkpoint(model, fused: bool):
    prefix = model.prefixes[0]
    experts = routed_expert_checkpoint(E, H, INTER, fused)
    return [(f"{prefix}.mlp.experts.{name}", tensor) for name, tensor in experts] + [
        (f"{prefix}.mlp.gate.weight", ROUTER)
    ]


def _reference(num_experts: int = E, seed: int = 0):
    reference = make_routed_experts(num_experts, H, INTER)
    list(
        reference.load_weights(
            routed_expert_checkpoint(num_experts, H, INTER, False, seed=seed)
        )
    )
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
    assert {
        f"{module}.mlp.experts.routed_experts.w13_weight",
        f"{module}.mlp.experts.routed_experts.w2_weight",
    } <= loaded


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("fused", [False, True])
def test_eplb_replicas_are_loaded(model_cls, fused):
    """With 4 redundant experts on EP rank 1 of 2, every local physical slot
    (replicas of logical experts 0-3 included) gets its logical expert."""
    ep = (2, 1)
    model = model_cls(fake_moe_layer(E, H, INTER, ep=ep))
    experts = _use_routed_experts(
        model, make_routed_experts(E, H, INTER, ep=ep, num_redundant_experts=4)
    )
    model.load_weights(_checkpoint(model, fused=fused))

    reference = _reference()
    # Physical experts 6-11 are local; 8-11 replicate logical experts 0-3.
    for local, logical in enumerate([6, 7, 0, 1, 2, 3]):
        assert torch.equal(experts.w13_weight[local], reference.w13_weight[logical])
        assert torch.equal(experts.w2_weight[local], reference.w2_weight[logical])


@pytest.mark.parametrize("model_cls", [_AmdModel, _AmdMTP])
@pytest.mark.parametrize("n_shared", [1, 2])
def test_fused_shared_experts_are_split_into_slots(model_cls, n_shared, monkeypatch):
    """ROCm AITER fused shared experts: the widened `mlp.shared_experts.*`
    tensor is split into `n_shared_experts` experts after the routed ones."""
    monkeypatch.setattr(
        ExpertMapManager, "_init_aiter_shared_experts_topK_buffer", lambda self: None
    )
    monkeypatch.setattr(
        moe_utils,
        "ExpertMapManager",
        lambda *args, **kwargs: ExpertMapManager(
            *args, **kwargs, num_fused_shared_experts=n_shared
        ),
    )
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True
    )
    model = model_cls(fake_moe_layer(E, H, INTER), n_shared_experts=n_shared)
    model.fused_shared_expert_layers = {0 if model_cls is _AmdModel else 1}
    experts = _use_routed_experts(model, make_routed_experts(E, H, INTER))
    assert experts.w13_weight.shape[0] == E + n_shared

    # One widened shared expert: gate/up rows and down columns of each slot.
    shared = dict(routed_expert_checkpoint(n_shared, H, INTER, False, seed=1))
    widened = {
        proj: torch.cat(
            [shared[f"{j}.{proj}_proj.weight"] for j in range(n_shared)],
            dim=1 if proj == "down" else 0,
        )
        for proj in ("gate", "up", "down")
    }
    prefix = model.prefixes[0]
    model.load_weights(
        _checkpoint(model, fused=False)
        + [
            (f"{prefix}.mlp.shared_experts.{proj}_proj.weight", tensor)
            for proj, tensor in widened.items()
        ]
    )

    # The references get fused shared slots too (still patched).
    reference = _reference()
    assert torch.equal(experts.w13_weight[:E], reference.w13_weight[:E])
    assert torch.equal(experts.w2_weight[:E], reference.w2_weight[:E])
    shared_reference = _reference(n_shared, seed=1)
    assert torch.equal(experts.w13_weight[E:], shared_reference.w13_weight[:n_shared])
    assert torch.equal(experts.w2_weight[E:], shared_reference.w2_weight[:n_shared])


@pytest.mark.parametrize("model_cls", [_NvidiaModel, _AmdModel])
def test_unmatched_expert_names_fall_through(model_cls):
    model = model_cls(fake_moe_layer(E, H, INTER))
    name = "layers.0.mlp.experts.7.w1.weight"
    with pytest.raises(KeyError):
        model.load_weights([(name, torch.zeros(INTER, H))])
