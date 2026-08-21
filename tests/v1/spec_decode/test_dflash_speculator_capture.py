# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for ``DFlashSpeculator.capture()`` (#53031).

The ``DFlashSpeculator.capture()`` log line ``"Capturing model for ... speculator"``
is the only externally visible signal for whether the drafter is captured.
Before the fix it printed unconditionally, so the line also appeared when
``init_cudagraph_manager`` resolved to ``CUDAGraphMode.NONE`` and the
capture loop ended up doing nothing. The fix guards the log on
``query_cudagraph_manager.needs_capture()`` and returns early when
nothing should be captured.

These tests inject a mock ``DFlashSpeculator`` so they can run on CPU
without a real CUDA graph setup.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def _make_speculator(needs_capture: bool) -> SimpleNamespace:
    """Build a hand-rolled stand-in for ``DFlashSpeculator`` that only
    exposes the attributes touched by ``capture()``.

    We bypass ``__init__`` because it requires a real ``VllmConfig`` and
    CUDA device. The class method under test (``capture``) reads exactly
    the attributes populated below, so the test is hermetic.
    """
    fake = SimpleNamespace(
        _speculator_name="DFlashTest",
        sample_indices=MagicMock(),
        sample_pos=MagicMock(),
        sample_idx_mapping=MagicMock(),
        _generate_draft=lambda *a, **k: None,
        input_buffers=MagicMock(),
        block_tables=MagicMock(),
        attn_groups=MagicMock(),
        kv_cache_config=MagicMock(),
        max_model_len=1024,
        _group_causal=False,
    )
    manager = MagicMock()
    manager.needs_capture.return_value = needs_capture
    fake.query_cudagraph_manager = manager
    return fake


def test_capture_skips_log_and_call_when_nothing_to_capture(caplog) -> None:
    """``needs_capture() == False`` ⇒ no log, no capture() call."""
    import logging

    speculator = _make_speculator(needs_capture=False)

    with caplog.at_level(logging.INFO, logger="vllm"):
        DFlashSpeculator.capture(speculator)

    msgs = [r.getMessage() for r in caplog.records]
    assert not any("Capturing model" in m for m in msgs), (
        f"unexpected capture log when needs_capture=False: {msgs!r}"
    )
    speculator.query_cudagraph_manager.capture.assert_not_called()


def test_capture_logs_and_runs_when_needed(caplog) -> None:
    """``needs_capture() == True`` ⇒ log line and capture() call both fire."""
    import logging

    speculator = _make_speculator(needs_capture=True)

    with caplog.at_level(logging.INFO, logger="vllm"):
        DFlashSpeculator.capture(speculator)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("Capturing model" in m and "DFlashTest" in m for m in msgs), (
        f"expected capture log when needs_capture=True: {msgs!r}"
    )
    speculator.query_cudagraph_manager.capture.assert_called_once()
