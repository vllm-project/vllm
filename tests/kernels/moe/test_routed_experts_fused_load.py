# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`RoutedExperts.load_weights` takes routed experts per expert
(`experts.E.{gate,up,down}_proj.weight`) and in the transformers v5 fused form
(`experts.gate_up_proj` [E, 2I, H], `experts.down_proj` [E, H, I]). Both forms,
and the fused fast path (one copy per shard and run of local slots), must
write the same bytes as the per-expert path, EPLB replicas included."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.model_loader.reload import layerwise

from .utils import make_routed_experts, routed_expert_checkpoint

E, H, INTER = 8, 16, 12
BLOCK = 2

# (expert TP (size, rank), EP (size, rank), placement, redundant experts)
PARALLEL = [
    ((1, 0), (1, 0), "linear", 0),
    ((2, 0), (1, 0), "linear", 0),
    ((2, 1), (1, 0), "linear", 0),
    ((1, 0), (2, 0), "linear", 0),
    ((1, 0), (4, 3), "linear", 0),
    ((1, 0), (4, 1), "round_robin", 0),
    ((2, 1), (2, 1), "linear", 0),
    ((1, 0), (1, 0), "linear", 4),
    ((1, 0), (2, 1), "linear", 4),
    ((2, 0), (2, 1), "linear", 4),
]


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _checkpoint(fused: bool, block_scales: bool) -> list[tuple[str, torch.Tensor]]:
    weights = routed_expert_checkpoint(E, H, INTER, fused)
    if not block_scales:
        return weights
    generator = torch.Generator().manual_seed(1)
    gate, up = (
        torch.randn(E, INTER // BLOCK, H // BLOCK, generator=generator)
        for _ in range(2)
    )
    down = torch.randn(E, H // BLOCK, INTER // BLOCK, generator=generator)
    if fused:
        return weights + [
            ("gate_up_proj_scale_inv", torch.cat([gate, up], dim=1)),
            ("down_proj_scale_inv", down),
        ]
    return weights + [
        (f"{e}.{proj}_proj.weight_scale_inv", scale[e])
        for e in range(E)
        for proj, scale in (("gate", gate), ("up", up), ("down", down))
    ]


def _make(tp, ep, placement, redundant, block_scales: bool) -> RoutedExperts:
    experts = make_routed_experts(
        E,
        H,
        INTER,
        tp=tp,
        ep=ep,
        placement=placement,
        num_redundant_experts=redundant,
    )
    if block_scales:
        # FP8 block scales; their loader branch depends only on the
        # parameter's quant_method and the name.
        num_local = experts.local_num_experts
        inter_per_rank = INTER // tp[0]
        for name, shape in (
            ("w13_weight_scale_inv", (2 * inter_per_rank // BLOCK, H // BLOCK)),
            ("w2_weight_scale_inv", (H // BLOCK, inter_per_rank // BLOCK)),
        ):
            param = torch.nn.Parameter(
                torch.empty(num_local, *shape), requires_grad=False
            )
            param.weight_loader = experts.weight_loader
            param.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            experts.register_parameter(name, param)
    for param in experts.parameters():
        param.data.fill_(-7.0)
    return experts


def _assert_same_bytes(a: RoutedExperts, b: RoutedExperts):
    params_b = dict(b.named_parameters())
    for name, param in a.named_parameters():
        assert torch.equal(
            param.data.view(torch.uint8), params_b[name].data.view(torch.uint8)
        ), name


def _count_shard_loads(monkeypatch, experts: RoutedExperts) -> list[int]:
    calls = []
    load = experts._load_model_weight_or_group_weight_scale

    def counting(**kwargs):
        calls.append(kwargs["expert_data"].shape[0])
        return load(**kwargs)

    monkeypatch.setattr(experts, "_load_model_weight_or_group_weight_scale", counting)
    return calls


@pytest.mark.parametrize("tp,ep,placement,redundant", PARALLEL)
@pytest.mark.parametrize("block_scales", [False, True])
def test_fused_and_per_expert_forms_write_the_same_bytes(
    monkeypatch, tp, ep, placement, redundant, block_scales
):
    args = (tp, ep, placement, redundant, block_scales)
    per_expert, fused, fused_slow = _make(*args), _make(*args), _make(*args)
    monkeypatch.setattr(fused_slow, "_covers_fused_load", lambda *_: False)

    loaded = set(per_expert.load_weights(_checkpoint(False, block_scales)))
    calls = _count_shard_loads(monkeypatch, fused)
    assert set(fused.load_weights(_checkpoint(True, block_scales))) == loaded
    assert set(fused_slow.load_weights(_checkpoint(True, block_scales))) == loaded

    _assert_same_bytes(per_expert, fused)
    _assert_same_bytes(per_expert, fused_slow)
    # Every local slot, EPLB replicas included, is written.
    assert torch.all(per_expert.w13_weight != -7.0)
    assert torch.all(per_expert.w2_weight != -7.0)

    num_shards = 6 if block_scales else 3
    assert sum(calls) == per_expert.local_num_experts * num_shards
    if placement == "linear" and redundant == 0:
        # One copy per shard.
        assert len(calls) == num_shards


def test_fused_fast_path_skips_copies_across_devices():
    experts = _make((1, 0), (1, 0), "linear", 0, block_scales=False)
    loaded = torch.empty(E, INTER, H, device="meta")
    assert not experts._covers_fused_load(experts.w13_weight, loaded, "w13_weight")


def test_fused_fast_path_skips_loaders_it_does_not_cover():
    experts = _make((1, 0), (1, 0), "linear", 0, block_scales=False)
    loaded = torch.empty(E, INTER, H)
    assert experts._covers_fused_load(experts.w13_weight, loaded, "w13_weight")
    experts.w13_weight.weight_loader = lambda *args, **kwargs: True
    assert not experts._covers_fused_load(experts.w13_weight, loaded, "w13_weight")


def test_unmatched_names_return_none():
    experts = _make((1, 0), (2, 1), "linear", 0, block_scales=False)
    # Matched, but expert 0 lives on EP rank 0.
    assert experts.load_expert_weight("0.gate_proj.weight", torch.zeros(6, H)) == []
    # Matches no mapping entry (e.g. checkpoint uses w1/w2/w3 names).
    assert experts.load_expert_weight("7.w1.weight", torch.zeros(6, H)) is None
    assert list(experts.load_weights([("7.w1.weight", torch.zeros(6, H))])) == []


@pytest.mark.parametrize(
    "tp,ep,redundant", [((1, 0), (1, 0), 0), ((1, 0), (2, 1), 4), ((2, 1), (2, 0), 4)]
)
def test_layerwise_reload_of_fused_names_completes_the_layer(
    monkeypatch, tp, ep, redundant
):
    """Layerwise reload wraps the loaders, so fused names take the per-expert
    fallback; the layer must count exactly load_numel_total and be processed
    while loading, EPLB replicas included."""
    reference, experts = (
        _make(tp, ep, "linear", redundant, block_scales=False) for _ in range(2)
    )
    list(reference.load_weights(_checkpoint(fused=False, block_scales=False)))
    processed = []
    process = layerwise._layerwise_process

    def recording(layer, info):
        processed.append(info.load_numel == info.load_numel_total)
        return process(layer, info)

    monkeypatch.setattr(layerwise, "_layerwise_process", recording)
    layerwise.record_metadata_for_reloading(experts)
    layerwise.initialize_layerwise_reload(experts)
    list(experts.load_weights(_checkpoint(fused=True, block_scales=False)))
    assert processed == [True]
    layerwise.finalize_layerwise_processing(experts, None)
    _assert_same_bytes(reference, experts)
