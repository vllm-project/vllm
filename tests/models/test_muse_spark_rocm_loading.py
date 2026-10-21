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


@pytest.mark.parametrize(
    "expert_parallel,reduce_dtype,supported",
    [(True, "fp32", True), (False, "bf16", True), (False, "fp32", False)],
)
def test_token_sharded_stream_needs_ep_or_a_bf16_tp_reduction(
    monkeypatch, expert_parallel: bool, reduce_dtype: str, supported: bool
) -> None:
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE", reduce_dtype)
    monkeypatch.setattr(envs, "VLLM_ROCM_MUSE_SP_MIN_TOKENS", 1024)
    parallel_config = SimpleNamespace(
        enable_expert_parallel=expert_parallel,
        tensor_parallel_size=8,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_sequence_parallel_moe=False,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(mlp_layer_types=["sparse"] * 4)
        ),
    )
    assert amd_model.token_sharded_stream_supported(vllm_config) is supported
