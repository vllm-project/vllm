# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BaseRenderer.warmup MM-warmup behavior.

These tests exercise:
  - Zero-limit modalities are filtered from mm_counts passed to
    get_dummy_processor_inputs (e.g. --limit-mm-per-prompt image=0 ...)
  - MM warmup is skipped entirely when mm_processor is None

No model weights are required: warmup() is called directly on a MagicMock
that acts as the renderer instance.
"""

from unittest.mock import MagicMock, patch

from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import ChatParams


def _make_renderer_mock(mm_limits: dict[str, int]) -> MagicMock:
    """Return a MagicMock that quacks like a BaseRenderer instance.

    render_chat is mocked to raise ChatTemplateResolutionError so the chat
    warmup block is skipped cleanly, keeping the test focused on MM warmup.
    """
    from vllm.entrypoints.chat_utils import ChatTemplateResolutionError

    renderer = MagicMock()

    # chat warmup: make render_chat raise so we skip past it cleanly
    renderer.render_chat.side_effect = ChatTemplateResolutionError("no template")

    # MM processor with configurable limits
    mm_processor = MagicMock()
    mm_processor.info.allowed_mm_limits = mm_limits
    renderer.mm_processor = mm_processor
    renderer._readonly_mm_processor = None
    renderer._warmup_mm_processor = BaseRenderer._warmup_mm_processor.__get__(
        renderer, BaseRenderer
    )
    renderer._clear_processor_cache = BaseRenderer._clear_processor_cache
    renderer.clear_mm_cache = MagicMock()
    renderer.model_config.max_model_len = 128
    renderer.model_config.get_multimodal_config.return_value.limit_per_prompt = {}

    return renderer


class TestMmWarmupZeroLimitFiltering:
    """Zero-limit modalities must be excluded from mm_counts."""

    def test_zero_limit_modality_excluded_from_mm_counts(self):
        """A modality with limit=0 must not appear in mm_counts."""
        renderer = _make_renderer_mock({"image": 1, "video": 0})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        get_inputs = renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs
        get_inputs.assert_called_once()
        _, kwargs = get_inputs.call_args
        assert "video" not in kwargs["mm_counts"]
        assert kwargs["mm_counts"]["image"] == 1

    def test_all_zero_limits_passes_empty_mm_counts(self):
        """When all limits are 0, mm_counts must be empty."""
        renderer = _make_renderer_mock({"image": 0, "video": 0})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        get_inputs = renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs
        get_inputs.assert_called_once()
        _, kwargs = get_inputs.call_args
        assert kwargs["mm_counts"] == {}

    def test_positive_limits_all_included_in_mm_counts(self):
        """All modalities with limit > 0 must be present in mm_counts."""
        renderer = _make_renderer_mock({"image": 2, "video": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        get_inputs = renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs
        get_inputs.assert_called_once()
        _, kwargs = get_inputs.call_args
        assert kwargs["mm_counts"] == {"image": 1, "video": 1}


class TestMmWarmupRunsNormally:
    """MM warmup must run when mm_processor is set and limits > 0."""

    def test_processor_apply_called(self):
        renderer = _make_renderer_mock({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_called_once()

    def test_mm_cache_cleared_after_warmup(self):
        renderer = _make_renderer_mock({"image": 1})

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.clear_mm_cache.assert_called_once()


class TestMmWarmupSkippedWhenNoProcessor:
    """MM warmup must be skipped when mm_processor is None (text-only model)."""

    def test_no_warmup_without_processor(self):
        renderer = _make_renderer_mock({})
        renderer.mm_processor = None  # override to None

        BaseRenderer.warmup(renderer, ChatParams())

        renderer.model_config.get_multimodal_config.assert_not_called()


class TestReadonlyMmWarmup:
    """Readonly MM processor warmup must mirror the render path behavior."""

    def test_readonly_processor_apply_called_and_cache_cleared(self):
        renderer = _make_renderer_mock({"image": 1})
        readonly_mm_processor = MagicMock()
        readonly_mm_processor.info.allowed_mm_limits = {"image": 1}
        renderer._readonly_mm_processor = readonly_mm_processor

        with patch("vllm.multimodal.processing.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        readonly_mm_processor.apply.assert_called_once()
        readonly_mm_processor.cache.clear_cache.assert_called_once()
