# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import pytest
import torch
import transformers
from transformers import AutoConfig, PreTrainedModel

from vllm.config import ModelConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.config import try_get_safetensors_metadata

from ..registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS


def get_tied_weight_names(model: PreTrainedModel) -> set[str]:
    """
    Get names of weights that are tied to other weights.
    Uses HuggingFace's internal _tied_weights_keys attribute.
    """
    tied_keys = getattr(model, "_tied_weights_keys", None)
    return set(tied_keys) if tied_keys else set()


def filter_tied_weights(
    weight_names: set[str],
    ref_weight_names: set[str],
    tied_weight_names: set[str],
) -> set[str]:
    """
    Remove tied weights that appear in named_parameters() but not in checkpoint.

    In transformers v5+, tied weights (e.g., lm_head tied to embed_tokens) appear
    separately in named_parameters() even though only one copy is stored in the
    checkpoint. _tied_weights_keys stores names relative to submodules, so we
    match by suffix (e.g., "lm_head.weight" matches "language_model.lm_head.weight").
    """
    if not tied_weight_names:
        return weight_names

    def is_tied_and_missing(name: str) -> bool:
        if name in ref_weight_names:
            return False
        return any(
            name == tied or name.endswith(f".{tied}") for tied in tied_weight_names
        )

    return {name for name in weight_names if not is_tied_and_missing(name)}


def create_repo_dummy_weights(repo: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Create weights from safetensors checkpoint metadata"""
    metadata = try_get_safetensors_metadata(repo)
    weight_names = list(metadata.weight_map.keys())
    with torch.device("meta"):
        return ((name, torch.empty(0)) for name in weight_names)


def create_dummy_model(repo: str, model_arch: str) -> PreTrainedModel:
    """
    Create weights from a dummy meta deserialized hf model with name conversion
    """
    model_cls: PreTrainedModel = getattr(transformers, model_arch)
    config = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        return model_cls._from_config(config)


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

    original_weights = create_repo_dummy_weights(model_id)
    hf_dummy_model = create_dummy_model(model_id, model_arch)
    hf_converted_weights = hf_dummy_model.named_parameters()
    hf_converted_buffers = hf_dummy_model.named_buffers()
    tied_weight_names = get_tied_weight_names(hf_dummy_model)
    mapper: WeightsMapper = model_cls.hf_to_vllm_mapper

    mapped_original_weights = mapper.apply(original_weights)
    mapped_hf_converted_weights = mapper.apply(hf_converted_weights)
    mapped_hf_converted_buffers = mapper.apply(hf_converted_buffers)

    ref_weight_names = set(map(lambda x: x[0], mapped_original_weights))
    weight_names = set(map(lambda x: x[0], mapped_hf_converted_weights))
    buffer_names = set(map(lambda x: x[0], mapped_hf_converted_buffers))
    ref_weight_names -= buffer_names

    # Filter out tied weights that don't exist in the checkpoint
    weight_names = filter_tied_weights(
        weight_names, ref_weight_names, tied_weight_names
    )

    weights_missing = ref_weight_names - weight_names
    weights_unmapped = weight_names - ref_weight_names
    assert not weights_missing and not weights_unmapped, (
        f"Following weights are not mapped correctly: {weights_unmapped}, "
        f"Missing expected weights: {weights_missing}."
    )
