# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ..models.utils import build_model_context


@pytest.mark.parametrize(
    "model_id,limit_mm_per_prompt,expected",
    [
        ("Qwen/Qwen2-0.5B-Instruct", {}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {}, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {
            "image": 0,
            "video": 0
        }, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {
            "image": 0
        }, True),
    ],
)
@pytest.mark.core_model
def test_supports_multimodal_inputs(model_id, limit_mm_per_prompt, expected):
    """Test supports_multimodal_inputs returns correct boolean for various 
    configs."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    assert MULTIMODAL_REGISTRY.supports_multimodal_inputs(
        ctx.model_config) is expected