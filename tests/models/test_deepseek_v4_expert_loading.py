# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 / V4.1 loaders send routed experts through
`load_routed_experts_weight`: per-expert (`experts.E.w{1,2,3}`) and fused
(`experts.w13` / `experts.w2`) names write what the layer's own
`RoutedExperts.load_weights` writes, EPLB replicas included."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.kernels.moe.utils import make_routed_experts, routed_expert_checkpoint
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.deepseek_v4.nvidia import dspark as v4_dspark
from vllm.models.deepseek_v4.nvidia import model as v4_model
from vllm.models.deepseek_v4.nvidia import mtp as v4_mtp
from vllm.models.deepseek_v41.nvidia import dspark as v41_dspark
from vllm.models.deepseek_v41.nvidia import model as v41_model

E, H, INTER = 8, 16, 12
CKPT_NAMES = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _ffn_layer(tp, ep, num_redundant_experts=0) -> nn.Module:
    """A decoder layer stand-in: `ffn.experts` is a `MoERunner` around a CPU
    `RoutedExperts` with DSv4's `w1/w2/w3` checkpoint names, params at -7."""
    experts = make_routed_experts(
        E, H, INTER, tp=tp, ep=ep, num_redundant_experts=num_redundant_experts
    )
    experts.ckpt_gate_proj_name = "w1"
    experts.ckpt_down_proj_name = "w2"
    experts.ckpt_up_proj_name = "w3"
    for param in experts.parameters():
        param.data.fill_(-7.0)
    runner = object.__new__(MoERunner)
    nn.Module.__init__(runner)
    runner.routed_experts = experts
    ffn = nn.Module()
    ffn.experts = runner
    ffn.gate = nn.Module()
    ffn.gate.weight = nn.Parameter(torch.zeros(E, H), requires_grad=False)
    ffn.use_native_mega_moe = False
    ffn.use_mega_moe = False
    ffn.n_redundant_experts = num_redundant_experts
    layer = nn.Module()
    layer.ffn = ffn
    return layer


def _main_model(model_cls):
    class Model(nn.Module):
        load_weights = model_cls.load_weights
        get_expert_mapping = model_cls.get_expert_mapping
        # (checkpoint prefix, module prefix) of the MoE layer
        prefixes = ("layers.0", "layers.0")

        def __init__(self, layer):
            super().__init__()
            self.config = SimpleNamespace(n_routed_experts=E)
            self.layers = nn.ModuleList([layer])
            self.start_layer, self.end_layer = 0, 1
            self.quant_config = None
            self.use_sequence_parallel = False

    return Model


def _mtp_model(model_cls):
    class Model(nn.Module):
        load_weights = model_cls.load_weights
        _to_spec_layer_name = model_cls._to_spec_layer_name
        _rewrite_spec_layer_name = model_cls._rewrite_spec_layer_name
        prefixes = ("mtp.0", "model.layers.1.mtp_block")

        def __init__(self, layer):
            super().__init__()
            self.config = SimpleNamespace(
                n_routed_experts=E,
                num_hidden_layers=1,
                num_nextn_predict_layers=1,
                expert_dtype="fp4",
            )
            self.quant_config = None
            self.pad_shared_expert = False
            block = nn.Module()
            block.mtp_block = layer
            self.model = nn.Module()
            self.model.layers = nn.ModuleDict({"1": block})
            self.model.mtp_start_layer_idx = 1
            self.model.num_mtp_layers = 1

    return Model


def _dspark_model(model_cls):
    class Model(nn.Module):
        load_weights = model_cls.load_weights
        _remap_dspark_name = model_cls._remap_dspark_name
        prefixes = ("mtp.0", "model.layers.0")

        def __init__(self, layer):
            super().__init__()
            self.config = SimpleNamespace(n_routed_experts=E, expert_dtype="fp4")
            self.quant_config = None
            self.pad_shared_expert = False
            self.linear_scale_name = "weight_scale_inv"
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
            self.model.confidence_head = None

    return Model


MODELS = {
    "v4": _main_model(v4_model.DeepseekV4Model),
    "v4_mtp": _mtp_model(v4_mtp.DeepSeekV4MTP),
    "v4_dspark": _dspark_model(v4_dspark.DSparkDeepseekV4ForCausalLM),
    "v41": _main_model(v41_model.DeepseekV4Model),
    "v41_dspark": _dspark_model(v41_dspark.DSparkDeepseekV4ForCausalLM),
}


def _expert_checkpoint(fused: bool) -> list[tuple[str, torch.Tensor]]:
    """Routed experts named relative to `experts`, in DSv4's names."""
    experts = []
    for name, tensor in routed_expert_checkpoint(E, H, INTER, fused):
        if fused:
            name = {"gate_up_proj": "w13", "down_proj": "w2"}[name]
        else:
            expert, proj, suffix = name.split(".", 2)
            name = f"{expert}.{CKPT_NAMES[proj]}.{suffix}"
        experts.append((name, tensor))
    return experts


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


def _checkpoint(model, fused: bool):
    prefix = model.prefixes[0]
    return [
        (f"{prefix}.ffn.experts.{name}", tensor)
        for name, tensor in _expert_checkpoint(fused)
    ] + [(f"{prefix}.ffn.gate.weight", ROUTER)]


def _experts(model) -> nn.Module:
    return model.get_submodule(f"{model.prefixes[1]}.ffn.experts").routed_experts


def _assert_same_experts(reference, model):
    for (name, a), b in zip(
        _experts(reference).named_parameters(), _experts(model).parameters()
    ):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name


@pytest.mark.parametrize("model_key", list(MODELS))
@pytest.mark.parametrize(
    "tp,ep,num_redundant_experts",
    [
        ((1, 0), (1, 0), 0),
        ((2, 1), (1, 0), 0),
        ((1, 0), (2, 1), 0),
        ((1, 0), (2, 1), 2),
    ],
)
def test_routed_experts_load_through_the_moe_layer(
    model_key, tp, ep, num_redundant_experts
):
    model_cls = MODELS[model_key]
    per_expert, fused, direct = (
        model_cls(_ffn_layer(tp, ep, num_redundant_experts)) for _ in range(3)
    )
    loaded = per_expert.load_weights(_checkpoint(per_expert, fused=False))
    fused.load_weights(_checkpoint(fused, fused=True))
    list(_experts(direct).load_weights(_expert_checkpoint(fused=False)))

    _assert_same_experts(direct, per_expert)
    _assert_same_experts(direct, fused)
    # Every local slot is written, EPLB replicas included.
    for param in _experts(per_expert).parameters():
        assert torch.all(param != -7.0)
    module = model_cls.prefixes[1]
    assert torch.equal(per_expert.get_submodule(f"{module}.ffn.gate").weight, ROUTER)
    assert loaded == {
        f"{module}.ffn.experts.routed_experts.w13_weight",
        f"{module}.ffn.experts.routed_experts.w2_weight",
        f"{module}.ffn.gate.weight",
    }


@pytest.mark.parametrize("model_key", ["v4", "v41"])
def test_unmatched_expert_names_raise(model_key):
    model = MODELS[model_key](_ffn_layer((1, 0), (1, 0)))
    with pytest.raises(KeyError):
        model.load_weights([("layers.0.ffn.experts.0.gate_proj.weight", ROUTER)])
