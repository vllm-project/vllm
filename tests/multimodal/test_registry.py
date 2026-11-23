# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget

pytestmark = pytest.mark.cpu_test


def test_create_processor_error_uses_served_model_name():
    model_config = SimpleNamespace(
        is_multimodal_model=False,
        model="/path/to/model/weights",
        served_model_name="friendly-model-name",
    )

    with pytest.raises(
        ValueError,
        match="friendly-model-name is not a multimodal model",
    ):
        MULTIMODAL_REGISTRY.create_processor(model_config)


def test_encoder_budget_excludes_modality_with_embeds_disabled():
    def encoder_cache_size(video_limit):
        model_config = ModelConfig(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            limit_mm_per_prompt={"image": 0, "video": video_limit},
            enable_mm_embeds=True,
        )
        vllm_config = VllmConfig(model_config=model_config)
        return MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY).encoder_cache_size

    image_only = encoder_cache_size({"count": 0, "allow_embeds": False})

    assert image_only < encoder_cache_size(0)
