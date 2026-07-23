# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import pytest
import torch
import transformers
from transformers import AutoConfig, AutoModel, PreTrainedModel

from vllm.config import ModelConfig
from vllm.model_executor.models.transformers.base import Base as TransformersBase
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.config import try_get_safetensors_metadata

from ..registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS


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


def create_repo_dummy_weights(repo: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Create weights from safetensors checkpoint metadata"""
    metadata = try_get_safetensors_metadata(repo)
    weight_names = list(metadata.weight_map.keys())
    with torch.device("meta"):
        return ((name, torch.empty(0)) for name in weight_names)


def create_dummy_base_model(repo: str, model_arch: str) -> PreTrainedModel:
    """
    Create weights from a dummy meta deserialized hf base model with name conversion
    """
    config = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    return model


def create_dummy_model(repo: str, model_arch: str) -> PreTrainedModel:
    """
    Create weights from a dummy meta deserialized hf model with name conversion
    """
    model_cls: PreTrainedModel = getattr(transformers, model_arch)
    config = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        model = model_cls._from_config(config)
    return model


def model_architectures_for_test() -> list[str]:
    arch_to_test = list[str]()
    for model_arch, info in _MULTIMODAL_EXAMPLE_MODELS.items():
        if not info.trust_remote_code and hasattr(transformers, model_arch):
            model_cls: PreTrainedModel = getattr(transformers, model_arch)
            if getattr(model_cls, "_checkpoint_conversion_mapping", None):
                arch_to_test.append(model_arch)
    return arch_to_test


@pytest.mark.core_model
@pytest.mark.parametrize("model_arch", model_architectures_for_test())
def test_hf_model_weights_mapper(model_arch: str):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    is_mistral_model = model_arch in [
        "Mistral3ForConditionalGeneration",
        "PixtralForConditionalGeneration",
        "VoxtralForConditionalGeneration",
    ]

    if not is_mistral_model or model_info.tokenizer_mode == "mistral":
        tokenizer_mode = model_info.tokenizer_mode
    else:
        tokenizer_mode = "hf"

    model_id = model_info.default

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=tokenizer_mode,
        config_format="hf",
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    if issubclass(model_cls, TransformersBase):
        # Transformers backend models create their mapper during __init__
        # by inspecting the HF model instance. We simulate this by calling
        # _create_hf_to_vllm_mapper with a minimal proxy object.
        model_cls = type(
            "ProxyModelCls",
            (),
            {
                "model": create_dummy_base_model(model_id, model_arch),
                "_maybe_apply_model_mapping": lambda self: None,
            },
        )()
        TransformersBase._create_hf_to_vllm_mapper(model_cls)

    original_weights = create_repo_dummy_weights(model_id)
    hf_dummy_model = create_dummy_model(model_id, model_arch)
    hf_converted_weights = hf_dummy_model.named_parameters()
    hf_converted_buffers = hf_dummy_model.named_buffers()
    mapper: WeightsMapper = model_cls.hf_to_vllm_mapper

    mapped_original_weights = mapper.apply(original_weights)
    mapped_hf_converted_weights = mapper.apply(hf_converted_weights)
    mapped_hf_converted_buffers = mapper.apply(hf_converted_buffers)

    ref_weight_names = set(map(lambda x: x[0], mapped_original_weights))
    weight_names = set(map(lambda x: x[0], mapped_hf_converted_weights))
    buffer_names = set(map(lambda x: x[0], mapped_hf_converted_buffers))

    # Some checkpoints may have buffers, we ignore them for this test
    ref_weight_names -= buffer_names

    # Some checkpoints include tied weights (e.g. lm_head tied to embed_tokens) in the
    # safetensors file. In Transformers v5, named_parameters() will not include them
    # after they are tied in the model, so the mapper will not be able to map them.
    # We exclude them from the reference weight names for this test.
    if isinstance(tied := getattr(hf_dummy_model, "_tied_weights_keys", None), dict):
        config = hf_dummy_model.config
        key = "tie_word_embeddings"
        if getattr(config.get_text_config(), key, False) or getattr(config, key, False):
            mapped_tied_weights = mapper.apply((k, None) for k in tied)
            tied_weight_names = set(map(lambda x: x[0], mapped_tied_weights))
            ref_weight_names -= tied_weight_names

    weights_missing = ref_weight_names - weight_names
    weights_unmapped = weight_names - ref_weight_names
    assert not weights_missing and not weights_unmapped, (
        f"Following weights are not mapped correctly: {weights_unmapped}, "
        f"Missing expected weights: {weights_missing}."
    )
