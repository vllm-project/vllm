# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that entry-point logits processor plugins correctly enable
output token id tracking.

Previously, ``gpu_model_runner.py`` derived the
``logitsprocs_need_output_token_ids`` flag solely from CLI-passed
``custom_logitsprocs``, ignoring entry-point plugins loaded by
``build_logitsprocs()``.  When the flag was ``False`` and all penalties
were neutral (``repetition_penalty=1.0``), vLLM's async scheduling path
filled the output token id buffer with ``-1`` placeholders instead of
real tokens, silently breaking any entry-point logits processor that
inspects generation history.

This test verifies that entry-point plugins cause
``LogitsProcessors.has_custom`` (and therefore the flag) to be ``True``.
"""

import importlib.metadata

import pytest
import torch

from tests.v1.logits_processors.utils import (
    DummyLogitsProcessor,
    entry_points as fake_entry_points,
)
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.logits_processor import build_logitsprocs

PIN_MEMORY_AVAILABLE = is_pin_memory_available()
DEVICE = current_platform.device_type


@pytest.fixture()
def _mock_entry_points(monkeypatch):
    """Inject a dummy non-argmax-invariant logits processor entry point."""
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


@pytest.mark.usefixtures("_mock_entry_points")
def test_entrypoint_plugin_enables_output_token_tracking():
    """Verify that a non-argmax-invariant entry-point plugin sets the
    output-token-tracking flag even when no CLI logits processors are
    provided.

    This is the scenario that was broken before the fix: a plugin like
    ``NoRepeatNGramLogitsProcessor`` loaded via ``vllm.logits_processors``
    entry-point group with ``is_argmax_invariant() = False`` would not
    cause ``logitsprocs_need_output_token_ids`` to be set, resulting in
    all ``-1`` placeholder tokens in the output buffer.
    """
    device = torch.device(DEVICE)
    vllm_config = VllmConfig()

    # No CLI-passed custom logits processors — only the entry-point plugin.
    custom_logitsprocs: tuple = ()

    logitsprocs = build_logitsprocs(
        vllm_config,
        device,
        PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
        custom_logitsprocs=custom_logitsprocs,
    )

    # The entry-point plugin (DummyLogitsProcessor) returns
    # is_argmax_invariant() = False, so it should appear in
    # non_argmax_invariant.
    entry_point_types = [
        type(p) for p in logitsprocs.non_argmax_invariant
    ]
    assert DummyLogitsProcessor in entry_point_types, (
        "DummyLogitsProcessor should be loaded via the entry-point mock"
    )

    # This is the key assertion: has_custom must be True when entry-point
    # plugins are present, even though custom_logitsprocs is empty.
    assert logitsprocs.has_custom is True, (
        "has_custom should be True when entry-point plugins are loaded, "
        "even with no CLI-passed custom_logitsprocs"
    )


@pytest.mark.usefixtures("_mock_entry_points")
def test_cli_logitsprocs_still_enable_tracking():
    """Verify that CLI-passed logits processors still enable tracking
    (existing behavior preserved)."""
    device = torch.device(DEVICE)
    vllm_config = VllmConfig()

    custom_logitsprocs = (DummyLogitsProcessor,)

    logitsprocs = build_logitsprocs(
        vllm_config,
        device,
        PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
        custom_logitsprocs=custom_logitsprocs,
    )

    assert logitsprocs.has_custom is True


def test_no_logitsprocs_disables_tracking():
    """Verify that has_custom is False when no custom logits processors
    are loaded (no CLI, no entry-point plugins)."""
    device = torch.device(DEVICE)
    vllm_config = VllmConfig()

    custom_logitsprocs: tuple = ()

    # Without mocking entry_points, no plugins will be found.
    logitsprocs = build_logitsprocs(
        vllm_config,
        device,
        PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
        custom_logitsprocs=custom_logitsprocs,
    )

    # Only builtins should be loaded — has_custom should be False.
    assert logitsprocs.has_custom is False, (
        "has_custom should be False when only builtins are loaded"
    )
