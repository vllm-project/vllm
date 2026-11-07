# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint name mapping and weight loading for the ROCm MuseSpark1x model."""

import importlib
from types import SimpleNamespace

import pytest
import torch

import vllm.models.muse_spark.amd.model as amd_model
from vllm.models.muse_spark.amd.model import (
    MuseSpark1xForCausalLM,
    MuseSpark1xModel,
)
from vllm.models.muse_spark.amd.multimodal import MuseSparkForConditionalGeneration
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkRoutedExperts,
    _track_quark_expert_weights,
    _validate_quark_expert_weights,
)
from vllm.models.muse_spark.common.moe import MuseSparkConsolidatedRoutedExperts
from vllm.platforms import current_platform


def test_rocm_mapper_retains_quark_dense_and_individual_expert_scales() -> None:
    """Quark scales survive the HF name mapping; q stacks into qkv_proj."""
    layer = "model.language_model.layers.0"
    expert_scale = f"{layer}.mlp.experts.7.gate_proj.weight_scale"
    dense_scale = f"{layer}.self_attn.o_proj.weight_scale"
    stacked_dense_scale = f"{layer}.self_attn.q_proj.weight_scale"

    amd_mapped = list(
        MuseSparkForConditionalGeneration.hf_to_vllm_mapper.apply(
            [
                (expert_scale, torch.ones(1)),
                (dense_scale, torch.ones(1)),
                (stacked_dense_scale, torch.ones(1)),
            ]
        )
    )

    assert [name for name, _ in amd_mapped] == [
        "model.layers.0.feed_forward.experts.7.gate_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.qkv_proj.weight_scale",
    ]
    assert amd_mapped[2][1].shard_id == "q"


def test_load_weights_moves_per_row_qk_scales_with_the_interleaved_rows(
    monkeypatch,
) -> None:
    """A PTPC-FP8 q/k projection dequantizes unchanged across the row interleave."""
    head_dim, q_rows, k_rows, hidden = 8, 4 * 8, 2 * 8, 16
    torch.manual_seed(0)
    weight = torch.randn(k_rows, hidden)
    scale = torch.rand(k_rows) + 0.5
    block_scale = torch.rand(k_rows, hidden // 4)
    input_scale = torch.tensor(0.25)
    # Same length as the k rows but indexed by input channel: must not move.
    per_channel_input = torch.rand(k_rows)
    o_proj = torch.randn(hidden, q_rows)
    layer = "model.language_model.layers.3.self_attn"
    seen: dict[str, torch.Tensor] = {}

    class RecordingLoader:
        def __init__(self, module) -> None:
            pass

        def load_weights(self, weights, *, mapper=None) -> set[str]:
            seen.update(dict(weights))
            return set(seen)

    model = object.__new__(MuseSpark1xForCausalLM)
    model.model = SimpleNamespace(
        config=SimpleNamespace(
            head_dim=head_dim, num_attention_heads=4, num_key_value_heads=2
        )
    )
    with monkeypatch.context() as patch:
        patch.setattr(amd_model, "AutoWeightsLoader", RecordingLoader)
        patch.setattr(
            amd_model, "materialize_residual_path_constants", lambda _model: 0
        )
        MuseSpark1xForCausalLM.load_weights(
            model,
            [
                (f"{layer}.k_proj.weight", weight),
                (f"{layer}.k_proj.weight_scale", scale),
                (f"{layer}.k_proj.weight_scale_block", block_scale),
                (f"{layer}.k_proj.input_scale", input_scale),
                (f"{layer}.k_proj.input_channel_scale", per_channel_input),
                (f"{layer}.o_proj.weight", o_proj),
            ],
        )

    def interleave(t: torch.Tensor) -> torch.Tensor:
        return (
            t.reshape(-1, 2, head_dim // 2, *t.shape[1:])
            .transpose(1, 2)
            .reshape(t.shape)
        )

    assert torch.equal(seen[f"{layer}.k_proj.weight"], interleave(weight))
    assert torch.equal(seen[f"{layer}.k_proj.weight_scale"], interleave(scale))
    assert torch.equal(
        seen[f"{layer}.k_proj.weight_scale_block"], interleave(block_scale)
    )
    assert torch.equal(
        seen[f"{layer}.k_proj.weight"] * seen[f"{layer}.k_proj.weight_scale"][:, None],
        interleave(weight * scale[:, None]),
    )
    assert seen[f"{layer}.k_proj.input_scale"] is input_scale
    assert seen[f"{layer}.k_proj.input_channel_scale"] is per_channel_input
    assert seen[f"{layer}.o_proj.weight"] is o_proj


def test_rocm_nested_mapper_preserves_individual_expert_projections() -> None:
    weights = [
        ("layers.0.feed_forward.experts.7.gate_proj.weight", torch.ones(1)),
        ("layers.0.feed_forward.experts.7.up_proj.weight", torch.ones(1)),
        ("layers.0.self_attn.q_proj.weight", torch.ones(1)),
    ]

    mapped = list(MuseSpark1xModel.hf_to_vllm_mapper.apply(weights))

    assert [name for name, _ in mapped] == [
        "layers.0.feed_forward.experts.7.gate_proj.weight",
        "layers.0.feed_forward.experts.7.up_proj.weight",
        "layers.0.self_attn.qkv_proj.weight",
    ]
    assert mapped[2][1].shard_id == "q"


def test_public_entry_point_selects_rocm_model(monkeypatch) -> None:
    import vllm.models.muse_spark as muse_spark

    original_model = muse_spark.MuseSpark1xForCausalLM
    original_omni_model = muse_spark.MuseSparkForConditionalGeneration
    with monkeypatch.context() as patch:
        patch.setattr(current_platform, "is_rocm", lambda: True)
        reloaded = importlib.reload(muse_spark)
        assert reloaded.MuseSpark1xForCausalLM.__module__.endswith(".amd.model")
        assert reloaded.MuseSparkForConditionalGeneration.__module__.endswith(
            ".amd.multimodal"
        )

    reloaded = importlib.reload(muse_spark)
    assert reloaded.MuseSpark1xForCausalLM is original_model
    assert reloaded.MuseSparkForConditionalGeneration is original_omni_model


def _individual_quark_expert_weights(
    num_experts: int,
) -> list[tuple[str, torch.Tensor]]:
    return [
        (f"{expert_id}.{projection}.{suffix}", torch.empty(1))
        for expert_id in range(num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale")
    ]


def test_quark_expert_weight_audit_accepts_fragmented_loader_calls() -> None:
    weights = _individual_quark_expert_weights(3)
    loaded: set[tuple[str, str, int]] = set()
    first = weights[:7]
    second = weights[7:]

    assert list(_track_quark_expert_weights(first, loaded)) == first
    assert list(_track_quark_expert_weights(second, loaded)) == second
    _validate_quark_expert_weights(loaded, 3)


@pytest.mark.parametrize(
    "missing_name",
    ["0.gate_proj.weight", "1.down_proj.weight_scale"],
)
def test_quark_expert_weight_audit_rejects_missing_tensor(missing_name: str) -> None:
    weights = [
        item for item in _individual_quark_expert_weights(2) if item[0] != missing_name
    ]
    loaded: set[tuple[str, str, int]] = set()
    list(_track_quark_expert_weights(weights, loaded))
    with pytest.raises(ValueError, match="Missing MuseSpark1x Quark expert tensors"):
        _validate_quark_expert_weights(loaded, 2)


def test_quark_expert_weight_audit_rejects_duplicate_across_loader_calls() -> None:
    weights = _individual_quark_expert_weights(1)
    loaded: set[tuple[str, str, int]] = set()
    list(_track_quark_expert_weights(weights[:1], loaded))
    with pytest.raises(ValueError, match="Duplicate MuseSpark1x expert tensor"):
        list(_track_quark_expert_weights(weights[:1], loaded))


def test_unquantized_loader_routes_per_expert_names_to_expert_mapping(
    monkeypatch,
) -> None:
    """Quark exports keep some layers in BF16 but still store them per expert."""
    consolidated_calls: list[list[str]] = []
    per_expert_calls: list[list[str]] = []

    def _consolidated(self, weights):
        consolidated_calls.append([name for name, _ in weights])
        return iter(())

    def _per_expert(self, weights):
        per_expert_calls.append([name for name, _ in weights])
        return iter(())

    monkeypatch.setattr(
        MuseSparkConsolidatedRoutedExperts, "load_weights", _consolidated
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts."
        "load_weights",
        _per_expert,
    )
    experts = MuseSparkRoutedExperts.__new__(MuseSparkRoutedExperts)
    object.__setattr__(experts, "quant_method", SimpleNamespace())

    stacked = [("w13_weight", torch.empty(1)), ("w2_weight", torch.empty(1))]
    list(MuseSparkRoutedExperts.load_weights(experts, stacked))
    assert consolidated_calls == [["w13_weight", "w2_weight"]]
    assert per_expert_calls == []

    individual = [
        (name, tensor)
        for name, tensor in _individual_quark_expert_weights(2)
        if name.endswith(".weight")
    ]
    list(MuseSparkRoutedExperts.load_weights(experts, individual))
    assert consolidated_calls == [["w13_weight", "w2_weight"]]
    assert per_expert_calls == [[name for name, _ in individual]]


def test_consolidated_loader_fills_every_initial_redundant_replica() -> None:
    """EPLB's initial placement maps physical expert p to logical p % num_logical."""
    num_logical, num_physical = 4, 6
    calls: list[tuple[str, int, float]] = []

    def weight_loader(*, param, loaded_weight, weight_name, shard_id, expert_id):
        calls.append((shard_id, expert_id, loaded_weight.item()))

    experts = MuseSparkConsolidatedRoutedExperts.__new__(
        MuseSparkConsolidatedRoutedExperts
    )
    object.__setattr__(
        experts,
        "moe_config",
        SimpleNamespace(num_experts=num_physical, num_logical_experts=num_logical),
    )
    object.__setattr__(
        experts, "w2_weight", SimpleNamespace(weight_loader=weight_loader)
    )

    w2 = torch.arange(num_logical, dtype=torch.float32).view(num_logical, 1)
    assert list(experts.load_weights([("w2_weight", w2)])) == ["w2_weight"]
    assert sorted(calls) == sorted(
        ("w2", physical, float(physical % num_logical))
        for physical in range(num_physical)
    )


def test_router_weight_reload_refolds_and_preserves_planes(monkeypatch) -> None:
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.models.muse_spark.amd.model import MuseSpark1xMoE

    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.router = torch.nn.Linear(4, 2, bias=False)
    moe.router.weight_loader = ReplicatedLinear.weight_loader.__get__(moe.router)
    moe.router.weight.weight_loader = moe._load_router_weight
    moe.router_output_multiplier = 0.25
    moe._router_multiplier_folded = False
    moe.moe_router = SimpleNamespace(router_output_multiplier=0.25)
    monkeypatch.setattr(amd_model, "_supports_router_bf16x3_device", lambda: True)
    monkeypatch.setattr(amd_model, "supports_router_bf16x3", lambda *args: True)

    weight_ptr = moe.router.weight.data_ptr()
    planes_ptr = None
    for value in (1.0, 3.0):
        raw = torch.full_like(moe.router.weight, value)
        moe.router.weight.weight_loader(moe.router.weight, raw)
        moe.fold_router_output_multiplier()
        expected = raw * 0.25
        torch.testing.assert_close(moe.router.weight, expected)
        torch.testing.assert_close(moe._router_planes.float().sum(0), expected)
        assert moe.router.weight.data_ptr() == weight_ptr
        if planes_ptr is None:
            planes_ptr = moe._router_planes.data_ptr()
        assert moe._router_planes.data_ptr() == planes_ptr
        assert moe.moe_router.router_output_multiplier is None

        # Repeated materialization without loading must not rescale the weights.
        moe.fold_router_output_multiplier()
        torch.testing.assert_close(moe.router.weight, expected)
        assert moe._router_planes.data_ptr() == planes_ptr


@pytest.mark.parametrize(
    "sharing, bypass", [(False, False), (True, False), (False, True)]
)
def test_aiter_kv_workspace_only_for_cache_readers(
    monkeypatch, default_vllm_config, sharing, bypass
) -> None:
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionImpl,
        AiterFlashAttentionMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.impl = AiterFlashAttentionImpl.__new__(AiterFlashAttentionImpl)
    layer.impl.sliding_window = None
    layer.impl.num_kv_heads = 1
    layer.impl.head_size = 64
    layer.kv_sharing_target_layer_name = "source" if sharing else None
    layer.kv_cache_update_bypass = bypass
    monkeypatch.setattr(
        default_vllm_config.compilation_config,
        "static_forward_context",
        {"attn": layer},
    )
    monkeypatch.setattr(
        default_vllm_config,
        "model_config",
        SimpleNamespace(
            dtype=torch.bfloat16,
            get_num_attention_heads=lambda _: 1,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 64,
        ),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    builder = AiterFlashAttentionMetadataBuilder(
        spec, ["attn"], default_vllm_config, torch.device("cpu")
    )
    assert (builder.kv_sharing_workspace is not None) == (sharing or bypass)
