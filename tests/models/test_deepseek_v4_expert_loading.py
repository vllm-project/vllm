# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 / V4.1 loaders send routed experts through
`load_routed_experts_weight`: per-expert (`experts.E.w{1,2,3}`) and fused
(`experts.w13` / `experts.w2`) names write what the layer's own
`RoutedExperts.load_weights` writes, EPLB replicas included."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.kernels.moe.utils import make_routed_experts, routed_expert_checkpoint
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    expert_map_manager,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.amd import dspark as v4_amd_dspark
from vllm.models.deepseek_v4.amd import model as v4_amd_model
from vllm.models.deepseek_v4.amd import mtp as v4_amd_mtp
from vllm.models.deepseek_v4.nvidia import dspark as v4_dspark
from vllm.models.deepseek_v4.nvidia import model as v4_model
from vllm.models.deepseek_v4.nvidia import mtp as v4_mtp
from vllm.models.deepseek_v41.amd import dspark as v41_amd_dspark
from vllm.models.deepseek_v41.amd import model as v41_amd_model
from vllm.models.deepseek_v41.nvidia import dspark as v41_dspark
from vllm.models.deepseek_v41.nvidia import model as v41_model

E, H, INTER = 8, 16, 12
CKPT_NAMES = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _ffn_layer(tp, ep, num_redundant_experts=0, num_shared=0) -> nn.Module:
    """A decoder layer stand-in: `ffn.experts` is a `MoERunner` around a CPU
    `RoutedExperts` with DSv4's `w1/w2/w3` checkpoint names, params at -7;
    `num_shared` fused shared experts follow the routed ones."""
    # The AITER shared-expert top-k buffer is a CUDA tensor the loader never reads.
    with patch.object(expert_map_manager, "init_aiter_topK_meta_data"):
        experts = make_routed_experts(
            E,
            H,
            INTER,
            tp=tp,
            ep=ep,
            num_redundant_experts=num_redundant_experts,
            num_fused_shared_experts=num_shared,
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
            self.config = SimpleNamespace(n_routed_experts=E, n_shared_experts=1)
            self.layers = nn.ModuleList([layer])
            self.start_layer, self.end_layer = 0, 1
            self.quant_config = None
            self.use_sequence_parallel = False
            self.use_mega_moe = False
            # AMD: layers whose shared expert is fused into the routed experts
            self.fused_shared_expert_layers: frozenset[int] = frozenset()

    return Model


def _mtp_model(model_cls):
    class Model(nn.Module):
        load_weights = model_cls.load_weights
        _to_spec_layer_name = getattr(model_cls, "_to_spec_layer_name", None)
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
    "v4_amd": _main_model(v4_amd_model.DeepseekV4Model),
    "v4_amd_mtp": _mtp_model(v4_amd_mtp.DeepSeekV4MTP),
    "v4_amd_dspark": _dspark_model(v4_amd_dspark.DSparkDeepseekV4ForCausalLM),
    "v41_amd": _main_model(v41_amd_model.DeepseekV4Model),
    "v41_amd_dspark": _dspark_model(v41_amd_dspark.DSparkDeepseekV4ForCausalLM),
}
# The NVIDIA MoE layers take EPLB redundant experts.
EPLB_MODELS = {"v4", "v4_mtp", "v4_dspark", "v41", "v41_dspark"}


def _expert_checkpoint(
    fused: bool, num_experts: int = E, seed: int = 0
) -> list[tuple[str, torch.Tensor]]:
    """Routed experts named relative to `experts`, in DSv4's names."""
    experts = []
    for name, tensor in routed_expert_checkpoint(
        num_experts, H, INTER, fused, seed=seed
    ):
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
    if num_redundant_experts and model_key not in EPLB_MODELS:
        pytest.skip("no EPLB on this platform's MoE layer")
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


@pytest.mark.parametrize("model_key", ["v4", "v41", "v4_amd", "v41_amd"])
def test_unmatched_expert_names_raise(model_key):
    model = MODELS[model_key](_ffn_layer((1, 0), (1, 0)))
    with pytest.raises(KeyError):
        model.load_weights([("layers.0.ffn.experts.0.gate_proj.weight", ROUTER)])


SHARED = [  # the one shared expert, as `shared_experts.w{1,2,3}.weight`
    (name.removeprefix("0."), tensor)
    for name, tensor in _expert_checkpoint(fused=False, num_experts=1, seed=2)
]


@pytest.mark.parametrize("model_key", ["v4_amd"])
@pytest.mark.parametrize("tp", [(1, 0), (2, 1)])
def test_fused_shared_expert_loads_into_the_appended_slot(model_key, tp):
    model = MODELS[model_key](_ffn_layer(tp, (1, 0), num_shared=1))
    model.fused_shared_expert_layers = frozenset({0})
    direct = MODELS[model_key](_ffn_layer(tp, (1, 0), num_shared=1))
    shared = [(f"layers.0.ffn.shared_experts.{n}", t) for n, t in SHARED]
    loaded = model.load_weights(_checkpoint(model, fused=False) + shared)
    list(
        _experts(direct).load_weights(
            _expert_checkpoint(fused=False) + [(f"{E}.{n}", t) for n, t in SHARED]
        )
    )

    _assert_same_experts(direct, model)
    for param in _experts(model).parameters():
        assert torch.all(param != -7.0)
    assert loaded == {
        "layers.0.ffn.experts.routed_experts.w13_weight",
        "layers.0.ffn.experts.routed_experts.w2_weight",
        "layers.0.ffn.gate.weight",
    }


def _add_block_scales(layer, suffix: str, block: int = 4) -> None:
    experts = layer.ffn.experts.routed_experts
    n_blocks, k_blocks = -(-INTER // block), -(-H // block)
    for name, shape in (
        ("w13", (E, 2 * n_blocks, k_blocks)),
        ("w2", (E, k_blocks, n_blocks)),
    ):
        param = nn.Parameter(torch.full(shape, -7.0), requires_grad=False)
        set_weight_attrs(
            param,
            {
                "weight_loader": experts.weight_loader,
                "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
            },
        )
        experts.register_parameter(f"{name}_{suffix}", param)


@pytest.mark.parametrize(
    "ckpt_suffix,param_suffix",
    [
        ("weight_scale_inv", "weight_scale"),
        ("weight_scale", "weight_scale_inv"),
        ("weight_scale_inv", "weight_scale_inv"),
    ],
)
def test_v41_amd_expert_scales_follow_the_registered_name(ckpt_suffix, param_suffix):
    """DSv4.1 AMD renames expert scales to the suffix the layer registers
    (FP8 `_inv` vs MXFP8 / Quark) before the MoE layer maps them."""
    model = MODELS["v41_amd"](_ffn_layer((1, 0), (1, 0)))
    _add_block_scales(model.layers[0], param_suffix)
    generator = torch.Generator().manual_seed(3)
    scales = {
        "w1": torch.rand(E, 3, 4, generator=generator),
        "w3": torch.rand(E, 3, 4, generator=generator),
        "w2": torch.rand(E, 4, 3, generator=generator),
    }
    loaded = model.load_weights(
        [
            (f"layers.0.ffn.experts.{e}.{w}.{ckpt_suffix}", scale[e])
            for w, scale in scales.items()
            for e in range(E)
        ]
    )

    experts = _experts(model)
    w13 = getattr(experts, f"w13_{param_suffix}")
    assert torch.equal(w13, torch.cat([scales["w1"], scales["w3"]], dim=1))
    assert torch.equal(getattr(experts, f"w2_{param_suffix}"), scales["w2"])
    prefix = "layers.0.ffn.experts.routed_experts"
    assert loaded == {f"{prefix}.w13_{param_suffix}", f"{prefix}.w2_{param_suffix}"}
