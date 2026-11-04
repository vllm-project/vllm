# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 (NVIDIA and AMD) and its MTP load routed experts through
`load_routed_experts_weight`: per-expert `w1/w2/w3` and transformers v5 fused
names write what the layer's own `RoutedExperts.load_weights` writes, also
behind a LoRA wrapper and, on AMD, with the shared expert fused into the
routed experts."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tests.kernels.moe.utils import (
    _LoRAWrappedMoE,
    fake_moe_layer,
    make_routed_experts,
    routed_expert_checkpoint,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3Model
from vllm.models.minimax_m3.nvidia.mtp import MiniMaxM3MTP

E, H, INTER = 8, 16, 12
CKPT_NAMES = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def _amd_modules():
    """The AMD model reads the ROCm arch at import; give it none on CPU."""
    with mock.patch(
        "torch.cuda.get_device_properties",
        return_value=SimpleNamespace(gcnArchName=""),
    ):
        from vllm.models.minimax_m3.amd import model, mtp
    return model, mtp


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _w123(name: str) -> str:
    """Rename a `routed_expert_checkpoint` name to MiniMax's w1/w2/w3."""
    for hf, ckpt in CKPT_NAMES.items():
        name = name.replace(f"{hf}.", f"{ckpt}.")
    return {"gate_up_proj": "w13", "down_proj": "w2"}.get(name, name)


def _checkpoint(fused: bool, suffix: str = "weight", num_experts: int = E):
    experts = routed_expert_checkpoint(num_experts, H, INTER, fused, suffix)
    return [(_w123(name), tensor) for name, tensor in experts]


def _moe_layer(
    tp, ep, lora=False, num_fused_shared_experts=0, num_redundant_experts=0
) -> nn.Module:
    """`fake_moe_layer` with MiniMax's `block_sparse_moe` and w1/w2/w3
    checkpoint names, optionally with fused shared experts or EPLB replicas."""
    layer = fake_moe_layer(E, H, INTER, tp, ep, lora)
    mlp = layer.mlp
    runner = getattr(mlp.experts, "base_layer", mlp.experts)
    if num_fused_shared_experts:
        runner.routed_experts = _fse_routed_experts(num_fused_shared_experts, ep)
    if num_redundant_experts:
        runner.routed_experts = make_routed_experts(
            E, H, INTER, tp, ep, num_redundant_experts=num_redundant_experts
        )
        for param in runner.routed_experts.parameters():
            param.data.fill_(-7.0)
    experts = runner.routed_experts
    experts.ckpt_gate_proj_name = "w1"
    experts.ckpt_down_proj_name = "w2"
    experts.ckpt_up_proj_name = "w3"
    del layer.mlp
    layer.block_sparse_moe = mlp
    return layer


def _fse_routed_experts(num_shared: int, ep) -> RoutedExperts:
    """A CPU `RoutedExperts` with `num_shared` fused shared experts appended
    after the routed ones (ROCm AITER FSE)."""
    with mock.patch.object(ExpertMapManager, "_init_aiter_shared_experts_topK_buffer"):
        reference = make_routed_experts(E, H, INTER, ep=ep)
        manager = ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=2,
            global_num_experts=E,
            num_redundant_experts=0,
            num_expert_group=2,
            moe_parallel_config=reference.moe_config.moe_parallel_config,
            placement_strategy="linear",
            enable_eplb=False,
            num_fused_shared_experts=num_shared,
            rocm_aiter_enabled=True,
        )
    moe_config = reference.moe_config
    moe_config.num_local_experts = manager.local_num_experts
    experts = RoutedExperts(
        "layers.0.block_sparse_moe.experts", torch.float32, moe_config, None, manager
    )
    for param in experts.parameters():
        param.data.fill_(-7.0)
    return experts


class _NvidiaModel(nn.Module):
    load_weights = MiniMaxM3Model.load_weights
    # (checkpoint prefix, module prefix) of the MoE layer
    prefixes = ("layers.0.block_sparse_moe", "layers.0.block_sparse_moe")

    def __init__(self, tp, ep, lora=False, num_redundant_experts=0):
        super().__init__()
        self.config = SimpleNamespace(num_local_experts=E)
        self.use_mega_moe = False
        self.layers = nn.ModuleList(
            [_moe_layer(tp, ep, lora, num_redundant_experts=num_redundant_experts)]
        )


class _NvidiaMTP(nn.Module):
    load_weights = MiniMaxM3MTP.load_weights
    _map_checkpoint_name = MiniMaxM3MTP._map_checkpoint_name
    _get_mtp_layer_idx_from_weight_name = (
        MiniMaxM3MTP._get_mtp_layer_idx_from_weight_name
    )
    _tracks_own_embed = False
    prefixes = (
        "language_model.model.mtp.layers.0.transformer_layer.block_sparse_moe",
        "model.layers.0.transformer_layer.block_sparse_moe",
    )

    def __init__(self, tp, ep, lora=False):
        super().__init__()
        self.config = SimpleNamespace(num_local_experts=E)
        self.model = nn.Module()
        self.model.num_mtp_layers = 1
        self.model.layers = nn.ModuleDict({"0": nn.Module()})
        self.model.layers["0"].transformer_layer = _moe_layer(tp, ep, lora)


def _model_classes():
    amd_model, amd_mtp = _amd_modules()

    class _AmdModel(_NvidiaModel):
        load_weights = amd_model.MiniMaxM3Model.load_weights

        def __init__(self, tp, ep, lora=False, fse=False):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(num_local_experts=E, n_shared_experts=1)
            self.fused_shared_expert_layers = {0} if fse else set()
            self.layers = nn.ModuleList(
                [_moe_layer(tp, ep, lora, num_fused_shared_experts=int(fse))]
            )

    class _AmdMTP(_NvidiaMTP):
        load_weights = amd_mtp.MiniMaxM3MTP.load_weights

    return {
        "nvidia": _NvidiaModel,
        "nvidia_mtp": _NvidiaMTP,
        "amd": _AmdModel,
        "amd_mtp": _AmdMTP,
    }


MODELS = ["nvidia", "nvidia_mtp", "amd", "amd_mtp"]


def _experts(model) -> RoutedExperts:
    runner = model.get_submodule(f"{model.prefixes[1]}.experts")
    return getattr(runner, "base_layer", runner).routed_experts


ROUTER = torch.randn(E, H, generator=torch.Generator().manual_seed(1))


def _model_checkpoint(model, fused: bool, suffix: str = "weight"):
    prefix = model.prefixes[0]
    return [
        (f"{prefix}.experts.{name}", tensor)
        for name, tensor in _checkpoint(fused, suffix)
    ] + [(f"{prefix}.gate.weight", ROUTER)]


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    "tp,ep", [((1, 0), (1, 0)), ((2, 1), (1, 0)), ((1, 0), (2, 1))]
)
def test_routed_experts_load_through_the_moe_layer(model_name, tp, ep):
    model_cls = _model_classes()[model_name]
    per_expert, fused, direct = (model_cls(tp, ep) for _ in range(3))
    loaded = per_expert.load_weights(_model_checkpoint(per_expert, fused=False))
    fused.load_weights(_model_checkpoint(fused, fused=True))
    list(_experts(direct).load_weights(_checkpoint(fused=False)))

    for model in (fused, direct):
        for (name, a), b in zip(
            _experts(per_expert).named_parameters(), _experts(model).parameters()
        ):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    assert torch.all(_experts(per_expert).w13_weight != -7.0)
    assert torch.all(_experts(per_expert).w2_weight != -7.0)
    module = model_cls.prefixes[1]
    assert torch.equal(per_expert.get_submodule(f"{module}.gate").weight, ROUTER)
    assert loaded == {
        f"{module}.experts.routed_experts.w13_weight",
        f"{module}.experts.routed_experts.w2_weight",
        f"{module}.gate.weight",
    }


@pytest.mark.parametrize("model_name", ["nvidia", "amd"])
def test_lora_wrapped_moe_layer_is_unwrapped(model_name):
    model = _model_classes()[model_name]((1, 0), (1, 0), lora=True)
    assert isinstance(
        model.get_submodule("layers.0.block_sparse_moe.experts"), _LoRAWrappedMoE
    )
    loaded = model.load_weights(
        _model_checkpoint(model, fused=False, suffix="base_layer.weight")
    )
    prefix = "layers.0.block_sparse_moe.experts.base_layer.routed_experts"
    assert loaded == {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
        "layers.0.block_sparse_moe.gate.weight",
    }
    assert loaded <= dict(model.named_parameters()).keys()
    reference = _model_classes()[model_name]((1, 0), (1, 0))
    reference.load_weights(_model_checkpoint(reference, fused=False))
    assert torch.equal(_experts(model).w13_weight, _experts(reference).w13_weight)
    assert torch.equal(_experts(model).w2_weight, _experts(reference).w2_weight)


@pytest.mark.parametrize("ep", [(1, 0), (2, 0), (2, 1)])
def test_amd_fused_shared_expert_loads_into_the_appended_slot(ep):
    """ROCm AITER FSE: the model renames `shared_experts.*` to expert
    `num_local_experts`, which the layer's own mapping covers."""
    model = _model_classes()["amd"]((1, 0), ep, fse=True)
    direct = _model_classes()["amd"]((1, 0), ep, fse=True)
    shared = routed_expert_checkpoint(1, H, INTER, fused=False, seed=3)
    checkpoint = _model_checkpoint(model, fused=False) + [
        (f"layers.0.block_sparse_moe.shared_experts.{name[2:]}", tensor)
        for name, tensor in shared
    ]
    with mock.patch(
        "vllm._aiter_ops.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
        return_value=True,
    ):
        loaded = model.load_weights(checkpoint)
    list(
        _experts(direct).load_weights(
            _checkpoint(fused=False)
            + [(f"{E}.{_w123(name[2:])}", tensor) for name, tensor in shared]
        )
    )
    assert loaded == {
        "layers.0.block_sparse_moe.experts.routed_experts.w13_weight",
        "layers.0.block_sparse_moe.experts.routed_experts.w2_weight",
        "layers.0.block_sparse_moe.gate.weight",
    }
    experts = _experts(model)
    assert torch.equal(experts.w13_weight, _experts(direct).w13_weight)
    assert torch.equal(experts.w2_weight, _experts(direct).w2_weight)
    # The shared expert is the last local slot on every rank.
    gate, up, down = (tensor for _, tensor in shared)
    assert torch.equal(experts.w13_weight[-1], torch.cat([gate, up]))
    assert torch.equal(experts.w2_weight[-1], down)
    assert torch.all(experts.w13_weight != -7.0)


def test_unmatched_expert_names_fall_through():
    """Without `ckpt_names=("w1", "w2", "w3")` on the layer the names match no
    mapping entry, are not loaded, and the loader skips them as before."""
    model = _NvidiaModel((1, 0), (1, 0))
    experts = _experts(model)
    experts.ckpt_gate_proj_name = "gate_proj"
    experts.ckpt_down_proj_name = "down_proj"
    experts.ckpt_up_proj_name = "up_proj"
    assert model.load_weights(_model_checkpoint(model, fused=False)) == {
        "layers.0.block_sparse_moe.gate.weight"
    }
    assert torch.all(experts.w13_weight == -7.0)


@pytest.mark.parametrize("ep", [(2, 0), (2, 1)])
def test_eplb_replicas_are_loaded(ep):
    """Contract check: through the MoE layer every physical slot, EPLB
    replicas included, is filled (the base loop's mapping had none). MiniMax-M3
    doesn't enable EPLB today, so this is not a runtime change."""
    per_expert, fused, direct = (
        _NvidiaModel((1, 0), ep, num_redundant_experts=2) for _ in range(3)
    )
    per_expert.load_weights(_model_checkpoint(per_expert, fused=False))
    fused.load_weights(_model_checkpoint(fused, fused=True))
    list(_experts(direct).load_weights(_checkpoint(fused=False)))
    for model in (fused, direct):
        assert torch.equal(_experts(per_expert).w13_weight, _experts(model).w13_weight)
        assert torch.equal(_experts(per_expert).w2_weight, _experts(model).w2_weight)
    assert torch.all(_experts(per_expert).w13_weight != -7.0)
    assert torch.all(_experts(per_expert).w2_weight != -7.0)


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_moe_layer_declares_w123_checkpoint_names(vendor):
    """The tests above set the checkpoint names on the fake layer; the real
    `MiniMaxM3MoE` must pass the same ones to `FusedMoEFactory`, or the
    layer's mapping matches no routed-expert name."""
    if vendor == "nvidia":
        from vllm.models.minimax_m3.nvidia import model as module
    else:
        module = _amd_modules()[0]
    config = SimpleNamespace(
        num_local_experts=E,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        n_shared_experts=0,
        hidden_size=H,
        intermediate_size=INTER,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    with (
        mock.patch.object(
            module, "get_tensor_model_parallel_world_size", return_value=1
        ),
        mock.patch.object(module, "GateLinear"),
        mock.patch.object(
            module, "FusedMoEFactory", return_value=nn.Module()
        ) as factory,
    ):
        module.MiniMaxM3MoE(config, layer_id=0, prefix="layers.0.block_sparse_moe")
    assert factory.call_args.kwargs["ckpt_names"] == ("w1", "w2", "w3")


@pytest.mark.parametrize("model_name", ["nvidia_mtp", "amd_mtp"])
def test_mtp_layer_with_only_remote_experts_counts_as_loaded(model_name):
    """Under EP an expert tensor of another rank matches the layer but loads
    nothing; it still marks the MTP layer as present, as the old loop did."""
    model = _model_classes()[model_name]((1, 0), (2, 1))
    name = f"{model.prefixes[0]}.experts.0.w1.weight"
    assert model.load_weights([(name, torch.zeros(INTER, H))]) == set()
    with pytest.raises(ValueError, match="Failed to load MTP layer"):
        model.load_weights([])


def test_mega_moe_experts_keep_the_model_mapping():
    """MegaMoE experts are not under a `MoERunner` (issue C): the helper
    passes them on and the model's own mega mapping loads them."""
    model = _NvidiaModel((1, 0), (1, 0))
    model.use_mega_moe = True
    experts = nn.Module()
    experts.w13_weight = nn.Parameter(torch.zeros(1), requires_grad=False)
    experts.w13_weight.weight_loader = mock.Mock()
    model.layers[0].block_sparse_moe.experts = experts
    model.get_expert_mapping = lambda: [("experts.w13_", "experts.3.w1.", 3, "w1")]
    weight = torch.ones(INTER, H)
    name = "layers.0.block_sparse_moe.experts.3.w1.weight"
    assert model.load_weights([(name, weight)]) == {
        "layers.0.block_sparse_moe.experts.w13_weight"
    }
    experts.w13_weight.weight_loader.assert_called_once_with(
        experts.w13_weight,
        weight,
        "layers.0.block_sparse_moe.experts.w13_weight",
        shard_id="w1",
        expert_id=3,
    )
