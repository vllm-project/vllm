# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint name mapping and weight loading for the CUDA MuseSpark1x model."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.models.muse_spark.common.layers import MuseSpark1xQKVOGateParallelLinear
from vllm.models.muse_spark.common.moe import MuseSparkConsolidatedRoutedExperts
from vllm.models.muse_spark.common.quant_config import (
    align_compressed_tensors_quant_config,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xForCausalLM,
    MuseSpark1xModel,
)
from vllm.models.muse_spark.nvidia.multimodal import MuseSparkForConditionalGeneration
from vllm.platforms import current_platform

_FP8_GROUP_ACTIVATIONS = {
    "dynamic": True,
    "group_size": 128,
    "num_bits": 8,
    "strategy": "group",
    "symmetric": True,
    "type": "float",
}

# llm-compressor's FP8-block dense + MXFP4 expert recipe for MuseSpark1x,
# written against the HF module names.
MXFP4_DSFP8_QUANTIZATION_CONFIG = {
    "config_groups": {
        "dense": {
            "format": "float-quantized",
            "input_activations": _FP8_GROUP_ACTIVATIONS,
            "targets": [
                r"re:.*self_attn\.(q|k|v|o|gate)_proj$",
                r"re:.*mlp\.(pre|post)_expert_proj$",
            ],
            "weights": {
                "block_structure": [128, 128],
                "dynamic": False,
                "num_bits": 8,
                "strategy": "block",
                "symmetric": True,
                "type": "float",
            },
        },
        "experts": {
            "format": "mxfp4-pack-quantized",
            "input_activations": _FP8_GROUP_ACTIVATIONS,
            "targets": [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"],
            "weights": {
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "scale_dtype": "torch.uint8",
                "strategy": "group",
                "symmetric": True,
                "type": "float",
            },
        },
    },
    "format": "mixed-precision",
    "ignore": [
        "lm_head",
        r"re:.*embed_tokens$",
        r"re:.*mlp\.gate$",
        r"re:model\.(audio_tower|vision_tower|vision_adapter|vision_projection).*",
    ],
    "quant_method": "compressed-tensors",
}


def _aligned_quant_config() -> CompressedTensorsConfig:
    config = CompressedTensorsConfig.from_config(
        copy.deepcopy(MXFP4_DSFP8_QUANTIZATION_CONFIG)
    )
    config.packed_modules_mapping = MuseSpark1xForCausalLM.packed_modules_mapping
    align_compressed_tensors_quant_config(config)
    return config


def test_regex_targets_follow_vllm_module_names() -> None:
    """Experts, projections and attention resolve to their checkpoint schemes."""
    config = _aligned_quant_config()
    module = torch.nn.Identity()
    layer = "model.layers.3"

    experts = config.get_scheme_dict(module, f"{layer}.feed_forward.experts.0.up_proj")
    assert experts is not None and experts["format"] == "mxfp4-pack-quantized"
    assert CompressedTensorsConfig._is_mxfp4_w4a8_fp8(
        experts["weights"], experts["input_activations"]
    )
    for name in ("feed_forward.pooler", "feed_forward.unpooler", "self_attn.o_proj"):
        dense = config.get_scheme_dict(module, f"{layer}.{name}")
        assert dense is not None and dense["weights"].strategy == "block", name
        assert CompressedTensorsConfig._is_fp8_w8a8(
            dense["weights"], dense["input_activations"]
        )
    # The fused Q/K/V(/O-gate) projection matches through packed_modules_mapping.
    assert config.get_scheme_dict(module, f"{layer}.self_attn.qkv_proj") is not None

    for name in (
        "lm_head",
        "model.embed_tokens",
        f"{layer}.feed_forward.router",
        "vision_model.encoder.layers.0.attn.qkv_proj",
    ):
        assert config.get_scheme_dict(module, name) is None, name


def test_quant_config_alignment_is_idempotent() -> None:
    config = _aligned_quant_config()
    targets, ignore = dict(config.target_scheme_map), list(config.ignore)
    align_compressed_tensors_quant_config(config)
    assert config.target_scheme_map == targets
    assert config.ignore == ignore


def test_cuda_mapper_keeps_per_expert_compressed_tensors_names() -> None:
    """Per-expert packed tensors stay unstacked; projections rename."""
    layer = "model.language_model.layers.0"
    mapped = list(
        MuseSparkForConditionalGeneration.hf_to_vllm_mapper.apply(
            [
                (f"{layer}.mlp.experts.7.gate_proj.weight_packed", torch.ones(1)),
                (f"{layer}.mlp.experts.7.up_proj.weight_scale", torch.ones(1)),
                (f"{layer}.mlp.pre_expert_proj.weight_scale", torch.ones(1)),
                (f"{layer}.mlp.post_expert_proj.weight", torch.ones(1)),
                (f"{layer}.self_attn.gate_proj.weight_scale", torch.ones(1)),
            ]
        )
    )
    assert [name for name, _ in mapped] == [
        "model.layers.0.feed_forward.experts.7.gate_proj.weight_packed",
        "model.layers.0.feed_forward.experts.7.up_proj.weight_scale",
        "model.layers.0.feed_forward.pooler.weight_scale",
        "model.layers.0.feed_forward.unpooler.weight",
        "model.layers.0.self_attn.qkv_proj.weight_scale",
    ]
    assert mapped[4][1].shard_id == "o_gate"


def test_cuda_model_mapper_keeps_expert_gate_up_unstacked() -> None:
    """LlamaModel's gate/up stacking must not fuse per-expert tensors.

    Stacked, ``experts.7.gate_proj`` becomes ``experts.7.gate_up_proj`` and
    RoutedExperts chunks the gate tensor in half as if it were fused.
    """
    experts = "layers.0.feed_forward.experts.7"
    mapped = list(
        MuseSpark1xModel.hf_to_vllm_mapper.apply(
            [
                (f"{experts}.gate_proj.weight_packed", torch.ones(1)),
                (f"{experts}.up_proj.weight_scale", torch.ones(1)),
                (f"{experts}.down_proj.weight_packed", torch.ones(1)),
                ("layers.0.self_attn.q_proj.weight", torch.ones(1)),
            ]
        )
    )
    assert [name for name, _ in mapped] == [
        f"{experts}.gate_proj.weight_packed",
        f"{experts}.up_proj.weight_scale",
        f"{experts}.down_proj.weight_packed",
        "layers.0.self_attn.qkv_proj.weight",
    ]
    assert all(getattr(tensor, "shard_id", None) is None for _, tensor in mapped[:3])
    assert mapped[3][1].shard_id == "q"


def test_consolidated_loader_routes_per_expert_names_to_expert_mapping(
    monkeypatch,
) -> None:
    """Per-expert tensors take RoutedExperts' path; weight_shape is dropped."""
    consolidated_calls: list[list[str]] = []
    per_expert_calls: list[list[str]] = []

    def _consolidated(self, weights):
        consolidated_calls.append([name for name, _ in weights])
        return iter(())

    def _per_expert(self, weights):
        per_expert_calls.append([name for name, _ in weights])
        return iter(())

    monkeypatch.setattr(
        MuseSparkConsolidatedRoutedExperts, "_load_consolidated_weights", _consolidated
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts."
        "load_weights",
        _per_expert,
    )
    experts = MuseSparkConsolidatedRoutedExperts.__new__(
        MuseSparkConsolidatedRoutedExperts
    )
    weights = [
        ("w13_weight", torch.empty(1)),
        ("7.gate_proj.weight_packed", torch.empty(1)),
        ("7.gate_proj.weight_shape", torch.empty(2)),
        ("7.down_proj.weight_scale", torch.empty(1)),
        ("w2_weight_scale", torch.empty(1)),
    ]
    list(experts.load_weights(weights))
    assert consolidated_calls == [["w13_weight", "w2_weight_scale"]]
    assert per_expert_calls == [
        [
            "7.gate_proj.weight_packed",
            "7.down_proj.weight_scale",
        ]
    ]


def test_qkv_o_gate_loader_slices_block_scales_by_scale_rows(dist_init) -> None:
    """An FP8 block scale shard lands on the rows of its weight shard."""
    head_dim, heads, kv_heads, hidden, tp_size, block_n = 128, 4, 2, 256, 2, 128
    layer = MuseSpark1xQKVOGateParallelLinear(
        hidden_size=hidden,
        head_size=head_dim,
        total_num_heads=heads,
        total_num_kv_heads=kv_heads,
        tp_rank=1,
        tp_size=tp_size,
    )
    layer.weight_block_size = [block_n, block_n]
    rows = layer.output_size // tp_size
    scale = BlockQuantScaleParameter(
        data=torch.zeros(rows // block_n, hidden // block_n),
        input_dim=1,
        output_dim=0,
        weight_loader=layer.weight_loader,
    )
    full_k_scale = torch.arange(1, 1 + kv_heads * head_dim // block_n).float()
    full_k_scale = full_k_scale[:, None].expand(-1, hidden // block_n)

    layer.weight_loader(scale, full_k_scale, "k")

    q_rows = heads // tp_size * head_dim
    expected = torch.zeros_like(scale.data)
    expected[q_rows // block_n] = 2.0  # rank 1's K head
    torch.testing.assert_close(scale.data, expected)

    with pytest.raises(ValueError, match="not aligned"):
        layer.weight_block_size = [256, 256]
        layer.weight_loader(scale, full_k_scale, "k")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_routed_experts_pick_hooked_mxfp4_for_compressed_tensors() -> None:
    """The dispatch lands on an MXFP4 kernel that runs the post-expert finalize:
    Humming where it outranks Marlin (Hopper), Marlin elsewhere."""
    from tests.kernels.moe.test_muse_spark_moe_deployment import _moe_config
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.quantization.utils.humming import prefers_humming
    from vllm.models.muse_spark.nvidia.muse_moe_runner import (
        MuseSparkCompressedTensorsMxfp4FusedMoE,
        MuseSparkRoutedExperts,
    )
    from vllm.utils.import_utils import has_humming

    config = _aligned_quant_config()
    moe_config = _moe_config()
    experts = MuseSparkRoutedExperts.__new__(MuseSparkRoutedExperts)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(experts, f"ckpt_{name}_name", name)
    experts.moe_config = moe_config
    with set_current_vllm_config(VllmConfig()):
        method = experts._get_quant_method(
            "model.layers.3.feed_forward.experts", config, moe_config
        )
    assert isinstance(method, MuseSparkCompressedTensorsMxfp4FusedMoE)
    if has_humming() and prefers_humming():
        assert method.experts_cls is HummingIndexedExperts
    else:
        assert method.experts_cls is MarlinExperts


def test_unsupported_quantization_is_rejected() -> None:
    quant_config = SimpleNamespace(get_name=lambda: "awq")
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
    )
    with pytest.raises(NotImplementedError, match="compressed-tensors"):
        MuseSpark1xForCausalLM.__init__(
            MuseSpark1xForCausalLM.__new__(MuseSpark1xForCausalLM),
            vllm_config=vllm_config,  # type: ignore[arg-type]
        )
