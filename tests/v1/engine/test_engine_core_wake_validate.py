# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for failed-wake state propagation in EngineCore.wake_up().

When the worker-level post-wake validation forward pass raises (a corrupt
wake), EngineCore.wake_up() must (a) mark the engine as fatally errored via the
_on_fatal_error hook so /health reports the truth, and (b) re-raise so the wake
call itself reports failure. A clean wake must do neither.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore


def _make_core(*, wake_side_effect=None):
    core = object.__new__(EngineCore)

    model_executor = MagicMock()
    if wake_side_effect is not None:
        model_executor.wake_up.side_effect = wake_side_effect
    core.model_executor = model_executor

    flags = {"resume_scheduler": 0, "on_fatal_error": 0}
    core.resume_scheduler = lambda: flags.__setitem__(
        "resume_scheduler", flags["resume_scheduler"] + 1
    )
    core._on_fatal_error = lambda: flags.__setitem__(
        "on_fatal_error", flags["on_fatal_error"] + 1
    )
    return core, model_executor, flags


def test_clean_wake_does_not_mark_dead():
    core, model_executor, flags = _make_core()

    EngineCore.wake_up(core, tags=None)

    model_executor.wake_up.assert_called_once_with(None)
    assert flags["on_fatal_error"] == 0
    assert flags["resume_scheduler"] == 1


def test_failed_wake_marks_engine_dead_and_reraises():
    """Pre-fix the exception escaped without marking the engine dead, so
    /health kept returning ready. Post-fix _on_fatal_error must fire AND the
    exception must propagate."""
    boom = RuntimeError("CUDA error: an illegal memory access was encountered")
    core, model_executor, flags = _make_core(wake_side_effect=boom)

    with pytest.raises(RuntimeError, match="illegal memory access"):
        EngineCore.wake_up(core, tags=None)

    assert flags["on_fatal_error"] == 1
    # Scheduling must NOT resume on a dead engine.
    assert flags["resume_scheduler"] == 0


def test_scheduling_only_wake_skips_executor():
    core, model_executor, flags = _make_core()

    EngineCore.wake_up(core, tags=["scheduling"])

    model_executor.wake_up.assert_not_called()
    assert flags["on_fatal_error"] == 0
    assert flags["resume_scheduler"] == 1


def test_base_on_fatal_error_is_noop():
    core = object.__new__(EngineCore)
    # Base hook must be a safe no-op (overridden by the multiproc core).
    assert EngineCore._on_fatal_error(core) is None
