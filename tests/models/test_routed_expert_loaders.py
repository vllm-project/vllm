# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-H MTP, MiMo-V2 and Kimi-K3 (model and MTP, NVIDIA and AMD) load
routed experts through `load_routed_experts_weight`, after each loader's own
name rewrites: per-expert and fused names write what the replaced
`make_expert_params_mapping` loop writes (every matching entry, so EPLB
replicas included) under TP, EP and EPLB redundant experts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.kernels.moe.utils import fake_moe_layer, routed_expert_checkpoint
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.models import mimo_v2
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.kimi_k3.amd import linear as kimi_amd
from vllm.models.kimi_k3.amd import mtp as kimi_amd_mtp
from vllm.models.kimi_k3.nvidia import model as kimi_nvidia
from vllm.models.kimi_k3.nvidia import mtp as kimi_nvidia_mtp

E, H, INTER = 8, 16, 12
ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


@pytest.fixture(autouse=True)
def _vllm_config(monkeypatch):
    monkeypatch.setattr(mimo_v2, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mimo_v2, "get_tensor_model_parallel_world_size", lambda: 1)
    with set_current_vllm_config(VllmConfig()):
        yield


class _Family(nn.Module):
    # checkpoint and module prefix of the MoE block and the block's attribute
    ckpt_prefix = module_prefix = "layers.0.mlp"
    block = "mlp"
    # (gate, down, up) checkpoint projection names, as the old loop mapped them
    ckpt_names: tuple[str, str, str | None] = ("gate_proj", "down_proj", "up_proj")
    # the fused gate/up checkpoint name
    fused_gate_up = "gate_up_proj"
    suffix = "weight"
    activation = MoEActivation.SILU

    def __init__(self, tp=(1, 0), ep=(1, 0), num_redundant_experts=0, pp=False):
        super().__init__()
        if pp:
            layer = PPMissingLayer()
        else:
            layer = fake_moe_layer(
                E,
                H,
                INTER,
                tp,
                ep,
                num_redundant_experts=num_redundant_experts,
                activation=self.activation,
                ckpt_names=self.ckpt_names,
            )
            if self.block != "mlp":
                setattr(layer, self.block, layer.mlp)
                del layer.mlp
        self.attach(layer)

    def experts(self) -> nn.Module:
        return self.get_submodule(f"{self.module_prefix}.experts").routed_experts

    @classmethod
    def expert_checkpoint(cls, fused: bool, suffix: str):
        """Expert tensors named relative to `experts` in this family's
        projection names; a non-gated family keeps only the up rows."""
        gate, down, up = cls.ckpt_names
        rename = {"gate_up_proj": cls.fused_gate_up, "down_proj": down}
        rename.update({"gate_proj": gate, "up_proj": up or gate})
        tensors = []
        for name, tensor in routed_expert_checkpoint(E, H, INTER, fused, suffix):
            proj = name.split(".")[-2] if "." in name else name
            if up is None and proj == "gate_proj":
                continue
            if up is None and proj == "gate_up_proj":
                tensor = tensor[:, INTER:]
            tensors.append((name.replace(proj, rename[proj]), tensor))
        return tensors

    def checkpoint(self, fused: bool):
        experts = self.expert_checkpoint(fused, self.suffix)
        return [
            (f"{self.ckpt_prefix}.experts.{name}", tensor) for name, tensor in experts
        ] + [(f"{self.ckpt_prefix}.gate.weight", ROUTER)]

    def load_with_old_loop(self, num_redundant_experts: int) -> None:
        """The replaced loop, trying every matching mapping entry."""
        mapping = fused_moe_make_expert_params_mapping(
            self, *self.ckpt_names, E, num_redundant_experts
        )
        params = dict(self.named_parameters())
        for name, tensor in self.expert_checkpoint(fused=False, suffix="weight"):
            name = f"{self.module_prefix}.experts.{name}"
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name in name:
                    mapped = name.replace(weight_name, param_name)
                    param = params[mapped]
                    param.weight_loader(
                        param, tensor, mapped, shard_id=shard_id, expert_id=expert_id
                    )


class _NemotronHMTP(_Family):
    load_weights = NemotronHMTP.load_weights
    is_unused_checkpoint_weight = NemotronHMTP.is_unused_checkpoint_weight
    ckpt_prefix = "language_model.mtp.layers.1.mixer"
    module_prefix = "model.layers.1.mixer"
    block = "mixer"
    ckpt_names = ("up_proj", "down_proj", None)
    fused_gate_up = "up_proj"
    activation = MoEActivation.RELU2_NO_MUL

    def attach(self, layer):
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({"1": layer})


class _MiMoV2(_Family):
    load_weights = mimo_v2.MiMoV2Model.load_weights
    _try_load_fp8_qkv_proj = mimo_v2.MiMoV2Model._try_load_fp8_qkv_proj

    def attach(self, layer):
        self.layers = nn.ModuleList([layer])


_KIMI_CONFIG = dict(
    linear_attn_config=None,
    q_lora_rank=None,
    is_moe=True,
    num_experts=E,
    is_linear_attn=False,
    num_hidden_layers=1,
)


class _KimiK3(_Family):
    load_weights = kimi_nvidia.KimiLinearModel.load_weights
    ckpt_prefix = module_prefix = "layers.0.block_sparse_moe"
    block = "block_sparse_moe"
    ckpt_names = ("w1", "w2", "w3")
    fused_gate_up = "w13"
    # compressed-tensors names; the loader renames them to the unpacked params
    suffix = "weight_packed"

    def attach(self, layer):
        self.config = SimpleNamespace(num_nextn_predict_layers=0, **_KIMI_CONFIG)
        self.layers = nn.ModuleList([layer])


class _KimiK3AMD(_KimiK3):
    load_weights = kimi_amd.KimiLinearModel.load_weights


class _KimiK3MTP(_KimiK3):
    load_weights = kimi_nvidia_mtp.KimiK3MTP.load_weights
    _rewrite_spec_layer_name = kimi_nvidia_mtp.KimiK3MTP._rewrite_spec_layer_name
    ckpt_prefix = "language_model.model.layers.1.block_sparse_moe"
    module_prefix = "model.layers.1.mtp_block.block_sparse_moe"

    def attach(self, layer):
        self.config = SimpleNamespace(num_nextn_predict_layers=1, **_KIMI_CONFIG)
        self.model = nn.Module()
        self.model.mtp_start_layer_idx = 1
        self.model.num_mtp_layers = 1
        self.model.layers = nn.ModuleDict({"1": nn.Module()})
        self.model.layers["1"].mtp_block = layer


class _KimiK3AMDMTP(_KimiK3MTP):
    load_weights = kimi_amd_mtp.KimiK3MTP.load_weights


FAMILIES = [_NemotronHMTP, _MiMoV2, _KimiK3, _KimiK3AMD, _KimiK3MTP, _KimiK3AMDMTP]
_ids = pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.__name__[1:])


@_ids
@pytest.mark.parametrize(
    "tp,ep,num_redundant_experts",
    [
        ((1, 0), (1, 0), 0),
        ((2, 1), (1, 0), 0),
        ((1, 0), (2, 1), 0),
        ((1, 0), (2, 1), 4),
    ],
    ids=["tp1", "tp2", "ep2", "ep2-eplb4"],
)
def test_routed_experts_load_through_the_moe_layer(
    family, tp, ep, num_redundant_experts
):
    if num_redundant_experts and issubclass(family, _KimiK3):
        pytest.skip("Kimi-K3 does not enable EPLB")
    per_expert, fused, old = (family(tp, ep, num_redundant_experts) for _ in range(3))
    loaded = per_expert.load_weights(per_expert.checkpoint(fused=False))
    fused.load_weights(fused.checkpoint(fused=True))
    old.load_with_old_loop(num_redundant_experts)

    for model in (fused, old):
        for (name, a), b in zip(
            per_expert.experts().named_parameters(),
            model.experts().parameters(),
            strict=True,
        ):
            assert torch.equal(a, b), name
    # Every local slot is written, EPLB replicas included.
    assert torch.all(per_expert.experts().w13_weight != -7.0)
    assert torch.all(per_expert.experts().w2_weight != -7.0)
    module = family.module_prefix
    assert torch.equal(per_expert.get_submodule(f"{module}.gate").weight, ROUTER)
    assert loaded == {
        f"{module}.experts.routed_experts.w13_weight",
        f"{module}.experts.routed_experts.w2_weight",
        f"{module}.gate.weight",
    }


@_ids
def test_unmatched_expert_names_fall_through(family):
    model = family()
    unknown = (f"{family.ckpt_prefix}.experts.7.unknown_proj.weight", torch.zeros(1))
    # The router keeps the MTP layer-completeness checks satisfied.
    weights = [unknown, (f"{family.ckpt_prefix}.gate.weight", ROUTER)]
    if family in (_KimiK3, _KimiK3AMD):
        with pytest.raises(KeyError):
            model.load_weights(weights)
    else:
        # These loaders skip names without a parameter, as on main.
        model.load_weights(weights)
    assert torch.all(model.experts().w13_weight == -7.0)


@_ids
def test_expert_names_without_a_param_raise(family):
    # The old loops skipped these (Nemotron-H MTP, MiMo-V2) or raised KeyError
    # (Kimi-K3), dropping e.g. scales the quant method did not register.
    name = f"{family.ckpt_prefix}.experts.7.{family.ckpt_names[0]}.input_scale"
    with pytest.raises(AttributeError, match="w13_input_scale"):
        family().load_weights([(name, torch.ones(1))])


@pytest.mark.parametrize(
    "family",
    [_NemotronHMTP, _MiMoV2, _KimiK3, _KimiK3AMD],
    ids=lambda f: f.__name__[1:],
)
def test_pp_missing_moe_layer_is_skipped(family):
    model = family(pp=True)
    for fused in (False, True):
        assert not model.load_weights(model.checkpoint(fused))
