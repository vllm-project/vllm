# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BaseRenderer.warmup MM-warmup skip conditions.

These tests exercise the guard added to skip multi-modal warmup when:
  - language_model_only=True, or
  - all allowed_mm_limits are 0 (e.g. --limit-mm-per-prompt image=0 ...)

No model weights are required: warmup() is called directly on a MagicMock
that acts as the renderer instance.
"""

from unittest.mock import MagicMock, patch

from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import ChatParams


def _make_renderer_mock(
    *,
    language_model_only: bool,
    mm_limits: dict[str, int],
) -> MagicMock:
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
    renderer.mm_processor = mm_processor  # truthy when mm_limits is non-empty

    # multimodal config
    mm_config = MagicMock()
    mm_config.language_model_only = language_model_only
    renderer.model_config.get_multimodal_config.return_value = mm_config

    return renderer


class TestMmWarmupSkippedWhenLanguageModelOnly:
    """MM warmup must not run when language_model_only=True."""

    def test_processor_apply_not_called(self):
        renderer = _make_renderer_mock(
            language_model_only=True,
            mm_limits={"image": 1, "video": 1},
        )

        BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_not_called()

    def test_dummy_inputs_not_fetched(self):
        renderer = _make_renderer_mock(
            language_model_only=True,
            mm_limits={"image": 1},
        )

        BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.dummy_inputs.get_dummy_processor_inputs.assert_not_called()


class TestMmWarmupSkippedWhenAllLimitsZero:
    """MM warmup must not run when all per-modality limits are 0."""

    def test_processor_apply_not_called_single_modality(self):
        renderer = _make_renderer_mock(
            language_model_only=False,
            mm_limits={"image": 0},
        )

        BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_not_called()

    def test_processor_apply_not_called_all_modalities(self):
        renderer = _make_renderer_mock(
            language_model_only=False,
            mm_limits={"image": 0, "video": 0, "audio": 0},
        )

        BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_not_called()


class TestMmWarmupRunsNormally:
    """MM warmup must run when language_model_only=False and limits > 0."""

    def test_processor_apply_called(self):
        renderer = _make_renderer_mock(
            language_model_only=False,
            mm_limits={"image": 1},
        )

        with patch("vllm.renderers.base.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_called_once()

    def test_processor_apply_called_mixed_limits(self):
        """At least one modality > 0 should trigger warmup."""
        renderer = _make_renderer_mock(
            language_model_only=False,
            mm_limits={"image": 1, "video": 0},
        )

        with patch("vllm.renderers.base.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.mm_processor.apply.assert_called_once()

    def test_mm_cache_cleared_after_warmup(self):
        renderer = _make_renderer_mock(
            language_model_only=False,
            mm_limits={"image": 1},
        )

        with patch("vllm.renderers.base.TimingContext", autospec=True):
            BaseRenderer.warmup(renderer, ChatParams())

        renderer.clear_mm_cache.assert_called_once()


class TestMmWarmupSkippedWhenNoProcessor:
    """MM warmup must be skipped when mm_processor is None (text-only model)."""

    def test_no_warmup_without_processor(self):
        renderer = _make_renderer_mock(
            language_model_only=False,
            mm_limits={},
        )
        renderer.mm_processor = None  # override to None

        BaseRenderer.warmup(renderer, ChatParams())

        # model_config.get_multimodal_config should never be called
        renderer.model_config.get_multimodal_config.assert_not_called()
