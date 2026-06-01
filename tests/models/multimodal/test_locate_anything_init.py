# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.registry import _MULTIMODAL_MODELS, ModelRegistry


def test_registered_in_model_registry():
    archs = ModelRegistry.get_supported_archs()
    assert "LocateAnythingForConditionalGeneration" in archs


def test_registry_mapping():
    assert _MULTIMODAL_MODELS["LocateAnythingForConditionalGeneration"] == (
        "locate_anything",
        "LocateAnythingForConditionalGeneration",
    )
