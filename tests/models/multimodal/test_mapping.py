# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import pytest
import torch
import transformers
from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig, PreTrainedModel

from vllm.config import ModelConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from ..registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS


def create_repo_dummy_weights(repo: str) -> Iterable[tuple[str, torch.Tensor]]:
    metadata = get_safetensors_metadata(repo)
    weight_names = list(metadata.weight_map.keys())
    with torch.device('meta'):
        return ((name, torch.empty(0)) for name in weight_names)


def get_dummy_loaded_hf_model(repo: str, model_arch: str) -> PreTrainedModel:
    model_cls: PreTrainedModel = getattr(transformers, model_arch)
    config = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        model: PreTrainedModel = model_cls._from_config(config)
    return model


@pytest.mark.parametrize("model_arch", _MULTIMODAL_EXAMPLE_MODELS.keys())
def test_hf_model_weights_mapper(model_arch: str):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if model_info.trust_remote_code:
        pytest.skip("Skip testing weights mapper for custom model")

    model_id = model_info.default
    dummy_hf_model = get_dummy_loaded_hf_model(model_id, model_arch)
    if not getattr(dummy_hf_model, "_checkpoint_conversion_mapping", None):
        pytest.skip("Skip HF models without checkpoint conversion mapping")

    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        seed=0,
        dtype="auto",
        revision=None,
        hf_overrides=model_info.hf_overrides,
    )
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)

    original_weights = create_repo_dummy_weights(model_id)
    hf_converted_weights = dummy_hf_model.named_parameters()
    mapper: WeightsMapper = model_cls.hf_to_vllm_mapper

    mapped_original_weights = mapper.apply(original_weights)
    mapped_hf_converted_weights = mapper.apply(hf_converted_weights)

    ref_weight_names = set(map(lambda x: x[0], mapped_original_weights))
    weight_names = set(map(lambda x: x[0], mapped_hf_converted_weights))
    assert ref_weight_names == weight_names
