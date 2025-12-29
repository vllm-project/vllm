# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ..models.utils import build_model_context

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "model_id,limit_mm_per_prompt,enable_mm_embeds,expected",
    [
        ("Qwen/Qwen2-0.5B-Instruct", {}, False, False),
        ("Qwen/Qwen2-0.5B-Instruct", {}, True, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {}, False, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {}, True, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0, "video": 0}, False, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0, "video": 0}, True, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0}, True, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0}, False, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 1, "video": 1}, True, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 1, "video": 1}, False, True),
    ],
)
@pytest.mark.core_model
def test_supports_multimodal_inputs(
    model_id, limit_mm_per_prompt, enable_mm_embeds, expected
):
    """Test supports_multimodal_inputs returns correct boolean for various
    configs.

    This test verifies that when enable_mm_embeds=True, the multimodal
    processing pipeline is enabled even when all modality limits are 0,
    allowing pre-computed embeddings to be processed while skipping
    encoder weight loading for memory optimization.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
        enable_mm_embeds=enable_mm_embeds,
    )
    assert MULTIMODAL_REGISTRY.supports_multimodal_inputs(ctx.model_config) is expected


@pytest.mark.core_model
def test_encoder_weight_skipping_with_enable_mm_embeds():
    """Integration test verifying encoder weights are not loaded when all
    modality limits are 0, even with enable_mm_embeds=True."""
    from vllm.v1.core.encoder_cache_manager import compute_encoder_cache_budget

    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_mm_embeds=True,
    )

    assert MULTIMODAL_REGISTRY.supports_multimodal_inputs(ctx.model_config) is True

    compute_budget, cache_size = compute_encoder_cache_budget(
        ctx.model_config, ctx.scheduler_config, MULTIMODAL_REGISTRY
    )
    assert compute_budget == 1, (
        "Compute budget should be minimal (1) for embeddings-only mode"
    )
    assert cache_size == 1, "Cache size should be minimal (1) for embeddings-only mode"