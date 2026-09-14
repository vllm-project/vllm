# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


def test_cosmos3_new_checkpoint_weights_mapper():
    from vllm.model_executor.models.cosmos3 import Cosmos3ForConditionalGeneration

    mapper = Cosmos3ForConditionalGeneration.hf_to_vllm_mapper

    assert mapper.apply_list(
        [
            "layers.0.self_attn.to_q.weight",
            "layers.0.self_attn.to_k.weight",
            "layers.0.self_attn.to_v.weight",
            "layers.0.self_attn.to_out.weight",
            "layers.0.self_attn.norm_q.weight",
            "layers.0.self_attn.norm_k.weight",
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
            "blocks.0.attn.qkv.weight",
        ]
    ) == [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.0.self_attn.k_norm.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        "visual.blocks.0.attn.qkv.weight",
    ]

    assert (
        mapper.apply_list(
            [
                "layers.0.self_attn.add_q_proj.weight",
                "layers.0.self_attn.add_k_proj.weight",
                "layers.0.self_attn.add_v_proj.weight",
                "layers.0.self_attn.to_add_out.weight",
                "layers.0.self_attn.norm_added_q.weight",
                "layers.0.self_attn.norm_added_k.weight",
                "layers.0.self_attn.q_proj_moe_gen.weight",
                "layers.0.mlp_moe_gen.gate_up_proj.weight",
                "norm_moe_gen.weight",
                "proj_in.weight",
                "proj_out.weight",
                "time_embedder.linear_1.weight",
                "audio_proj_in.weight",
                "audio_proj_out.weight",
                "action_proj_in.weight",
                "action_proj_out.weight",
                "audio_modality_embed",
                "action_modality_embed",
            ]
        )
        == []
    )


def test_cosmos3_modelopt_quantizer_weights_mapper():
    """ModelOpt/Diffusers FP8 checkpoints ship native fake-quant buffers
    (``*_quantizer._amax`` / ``._scale``) alongside the vLLM-consumable
    ``weight_scale`` / ``input_scale`` sidecars. vLLM must drop the former
    (it has no parameter for them) while keeping the latter."""
    from vllm.model_executor.models.cosmos3 import Cosmos3ForConditionalGeneration

    mapper = Cosmos3ForConditionalGeneration.hf_to_vllm_mapper

    # Native ModelOpt quantizer buffers are dropped.
    assert (
        mapper.apply_list(
            [
                "layers.0.self_attn.to_q.input_quantizer._amax",
                "layers.0.self_attn.to_q.weight_quantizer._amax",
                "layers.0.self_attn.to_q.weight_quantizer._scale",
                "layers.0.mlp.down_proj.output_quantizer._amax",
            ]
        )
        == []
    )

    # The FP8 scale sidecars vLLM actually consumes are kept and remapped.
    assert mapper.apply_list(
        [
            "layers.0.self_attn.to_q.weight",
            "layers.0.self_attn.to_q.weight_scale",
            "layers.0.self_attn.to_q.input_scale",
            "layers.0.mlp.down_proj.input_scale",
        ]
    ) == [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight_scale",
        "language_model.model.layers.0.self_attn.q_proj.input_scale",
        "language_model.model.layers.0.mlp.down_proj.input_scale",
    ]


def test_cosmos3_edge_checkpoint_weights_mapper():
    from vllm.model_executor.models.cosmos3_edge import (
        Cosmos3EdgeForConditionalGeneration,
    )

    mapper = Cosmos3EdgeForConditionalGeneration.hf_to_vllm_mapper

    assert mapper.apply_list(
        [
            "embed_tokens.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.self_attn.to_q.weight",
            "layers.0.self_attn.to_k.weight",
            "layers.0.self_attn.to_v.weight",
            "layers.0.self_attn.to_out.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.27.input_layernorm.weight",
            "layers.27.self_attn.to_q.weight",
            "layers.27.post_attention_layernorm.weight",
            "layers.27.mlp.down_proj.weight",
            "model.visual.embeddings.patch_embedding.weight",
            "model.visual.encoder.layers.0.self_attn.q_proj.weight",
            "model.projector.linear_fc1.weight",
            "lm_head.weight",
        ]
    ) == [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm_f.weight",
        "language_model.model.layers.0.norm.weight",
        "language_model.model.layers.0.mixer.qkv_proj.weight",
        "language_model.model.layers.0.mixer.qkv_proj.weight",
        "language_model.model.layers.0.mixer.qkv_proj.weight",
        "language_model.model.layers.0.mixer.o_proj.weight",
        "language_model.model.layers.1.norm.weight",
        "language_model.model.layers.1.mixer.up_proj.weight",
        "language_model.model.layers.1.mixer.down_proj.weight",
        "language_model.model.layers.54.norm.weight",
        "language_model.model.layers.54.mixer.qkv_proj.weight",
        "language_model.model.layers.55.norm.weight",
        "language_model.model.layers.55.mixer.down_proj.weight",
        "visual.encoder.embeddings.patch_embedding.weight",
        "visual.encoder.encoder.layers.0.self_attn.qkv_proj.weight",
        "visual.projector.linear_fc1.weight",
        "language_model.lm_head.weight",
    ]

    assert (
        mapper.apply_list(
            [
                "layers.0.self_attn.add_q_proj.weight",
                "layers.0.self_attn.add_k_proj.weight",
                "layers.0.self_attn.add_v_proj.weight",
                "layers.0.self_attn.to_add_out.weight",
                "layers.0.self_attn.norm_added_q.weight",
                "layers.0.self_attn.norm_added_k.weight",
                "layers.0.self_attn.k_norm_und_for_gen.weight",
                "layers.0.self_attn.q_proj_moe_gen.weight",
                "layers.0.mlp_moe_gen.up_proj.weight",
                "norm_moe_gen.weight",
                "proj_in.weight",
                "proj_out.weight",
                "audio_modality_embed",
                "action_modality_embed",
            ]
        )
        == []
    )


@pytest.mark.cpu_test
def test_bailing_vl_mapper_handles_module_and_parameter_names():
    from vllm.model_executor.models.bailing_moe_v3_vl import (
        BailingMoeV3VLForConditionalGeneration,
    )

    mapper = BailingMoeV3VLForConditionalGeneration.hf_to_vllm_mapper
    names = {
        "model.visual": "visual",
        "model.visual.blocks.0.attn.qkv.weight": "visual.blocks.0.attn.qkv.weight",
        "model.linear_proj": "linear_proj",
        "linear_proj.0": "linear_proj.linear_fc1",
        "model.linear_proj.0.weight": "linear_proj.linear_fc1.weight",
        "linear_proj.2": "linear_proj.linear_fc2",
        "linear_proj.2.weight_scale": "linear_proj.linear_fc2.weight_scale",
        "model.linear_proj.2.bias": "linear_proj.linear_fc2.bias",
        "linear_proj.linear_fc1.weight": "linear_proj.linear_fc1.weight",
        "model.linear_proj.linear_fc2.weight": "linear_proj.linear_fc2.weight",
        "model.layers.0.attention.q_proj.weight": (
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ),
        "model.layers.0.mlp.experts.3.up_proj.weight": (
            "language_model.model.layers.0.mlp.experts.3.up_proj.weight"
        ),
        "lm_head": "language_model.lm_head",
        "lm_head.weight": "language_model.lm_head.weight",
    }
    assert mapper.apply_list(list(names)) == list(names.values())
