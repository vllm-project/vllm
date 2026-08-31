# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v4.nvidia.vl_model import (
    _make_deepseek_v4_vl_weights_mapper,
)


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_vl_weights_mapper_reroots_text_weights(expert_dtype):
    mapper = _make_deepseek_v4_vl_weights_mapper(expert_dtype, image_enabled=True)
    apply = mapper._map_name

    assert apply("layers.0.ffn.gate.bias") == (
        "language_model.model.layers.0.ffn.gate.e_score_correction_bias"
    )
    assert apply("layers.7.ffn.gate.bias_vl") == (
        "language_model.model.layers.7.ffn.gate.bias_vl"
    )
    assert apply("head.weight") == "language_model.lm_head.weight"
    assert apply("embed.weight") == "language_model.model.embed_tokens.weight"
    assert apply("norm.weight") == "language_model.model.norm.weight"
    assert apply("hc_head_fn") == "language_model.model.hc_head_fn"
    assert apply("layers.0.ffn.shared_experts.w2.weight") == (
        "language_model.model.layers.0.ffn.shared_experts.down_proj.weight"
    )

    expert_scale = apply("layers.0.ffn.experts.3.w1.scale")
    if expert_dtype == "fp4":
        assert expert_scale == (
            "language_model.model.layers.0.ffn.experts.3.w1.weight_scale"
        )
    else:
        assert expert_scale == (
            "language_model.model.layers.0.ffn.experts.3.w1.weight_scale_inv"
        )


def test_vl_weights_mapper_vision_weights_passthrough():
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", image_enabled=True)
    apply = mapper._map_name

    assert apply("vision.blocks.0.attn.wqkv.weight") == (
        "vision.blocks.0.attn.wqkv.weight"
    )
    assert apply("vision.patch_embed.proj.bias") == "vision.patch_embed.proj.bias"
    assert apply("vision.norm.weight") == "vision.norm.weight"
    assert apply("aligner.w1.weight") == "aligner.w1.weight"
    assert apply("image_start") == "image_start"
    assert apply("image_pad") == "image_pad"


def test_vl_weights_mapper_drops_mtp_weights():
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", image_enabled=True)

    assert mapper._map_name("mtp.0.ffn.gate.bias_vl") is None
    assert mapper._map_name("mtp.0.hc_attn_base") is None


def test_vl_weights_mapper_drops_tower_when_image_disabled():
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", image_enabled=False)

    assert mapper._map_name("vision.blocks.0.attn.wqkv.weight") is None
    assert mapper._map_name("aligner.w1.weight") is None
    assert mapper._map_name("image_start") is None
    # bias_vl still loads: the MoE gate keeps it whenever vision_n_layers > 0
    assert mapper._map_name("layers.7.ffn.gate.bias_vl") is not None
