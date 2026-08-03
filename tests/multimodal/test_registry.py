# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

from types import SimpleNamespace

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ..models.utils import build_model_context

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "model_id,limit_mm_per_prompt,expected",
    [
        ("Qwen/Qwen2-0.5B-Instruct", {}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {}, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0, "video": 0}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0}, True),
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
    assert MULTIMODAL_REGISTRY.supports_multimodal_inputs(ctx.model_config) is expected


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


@pytest.mark.parametrize(
    "runner_type,expect_warning",
    [
        # A model that is served directly has no business declaring
        # SupportsMultiModal without a processor -- keep warning about it.
        ("generate", True),
        # Speculative drafters for multimodal targets legitimately do, so
        # text-only mode is expected and should stay quiet.
        ("draft", False),
    ],
)
def test_missing_processor_falls_back_to_text_only(
    monkeypatch, caplog, runner_type, expect_warning
):
    # The model path doubles as the `warning_once` cache key, so give each
    # case its own to keep them independent.
    model_config = SimpleNamespace(
        is_multimodal_model=True,
        model=f"/path/to/{runner_type}/weights",
        runner_type=runner_type,
        get_multimodal_config=SimpleNamespace,
    )

    def _no_processor(*args, **kwargs):
        raise ValueError("Model class SomeMTP has no registered multimodal processor")

    monkeypatch.setattr(MULTIMODAL_REGISTRY, "_create_processing_info", _no_processor)

    with caplog.at_level("WARNING", logger="vllm.multimodal.registry"):
        assert MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config) is False

    warned = "has no registered multimodal processor" in caplog.text
    assert warned is expect_warning
