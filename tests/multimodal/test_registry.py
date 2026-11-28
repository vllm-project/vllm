# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig, ModelConfig, VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
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


@pytest.mark.parametrize("consumer_role", ["ec_consumer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("image_limit", [0, 1])
def test_consumer_config_preserves_text_only_multimodal_limits(
    consumer_role, image_limit
):
    """A consumer accepts embedding metadata only when image inputs are enabled."""
    model_config = ModelConfig(
        "mistralai/Pixtral-12B-2409",
        tokenizer_mode="mistral",
        config_format="mistral",
        max_model_len=4096,
        limit_mm_per_prompt={"image": image_limit},
    )
    ec_config = (
        ECTransferConfig(ec_connector="ECCPUConnector", ec_role="ec_consumer")
        if consumer_role == "ec_consumer"
        else None
    )
    kv_config = None
    if consumer_role != "ec_consumer":
        kv_config = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer" if consumer_role == "kv_consumer" else "kv_both",
        )
    config = VllmConfig(
        model_config=model_config,
        ec_transfer_config=ec_config,
        kv_transfer_config=kv_config,
    )
    mm_config = model_config.get_multimodal_config()
    assert model_config.supports_multimodal_inputs is bool(image_limit)
    assert mm_config.enable_mm_embeds is bool(image_limit)
    assert mm_config.allow_missing_mm_embeddings is bool(image_limit)
    if image_limit:
        budget = MultiModalBudget(config, MULTIMODAL_REGISTRY)
        assert budget.encoder_cache_size > 0
