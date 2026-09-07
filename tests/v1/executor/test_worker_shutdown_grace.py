# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the worker shutdown grace on ROCm.

The executor escalates to SIGKILL after
``worker_shutdown_grace_s() + WORKER_SIGTERM_GRACE_S``. On ROCm the EngineCore
process manager separately allows ``ROCM_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S``
for device teardown, so the executor's schedule must not expire first, or the
workers holding the device memory are killed inside the window that exists to
protect them.
"""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.v1.engine.utils import ROCM_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S
from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import (
    WORKER_SIGTERM_GRACE_S,
    worker_shutdown_grace_s,
)

ENV = "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"


def _grace(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_rocm: bool,
    explicit: str | None = None,
    configured: int = 5,
) -> float:
    monkeypatch.setattr(envs, ENV, configured)
    monkeypatch.setattr(
        multiproc_executor,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: is_rocm),
    )
    if explicit is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, explicit)
    return worker_shutdown_grace_s()


def test_rocm_default_does_not_expire_before_engine_process_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the fix: the inner schedule must not preempt the outer."""
    grace = _grace(monkeypatch, is_rocm=True)
    assert grace + WORKER_SIGTERM_GRACE_S >= ROCM_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S


def test_rocm_default_is_stretched_above_the_generic_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _grace(monkeypatch, is_rocm=True) > 5.0


def test_non_rocm_default_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _grace(monkeypatch, is_rocm=False) == 5.0


def test_explicit_setting_is_honoured_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user who asks for a short grace gets it, even on ROCm."""
    assert _grace(monkeypatch, is_rocm=True, explicit="2", configured=2) == 2.0


def test_explicit_setting_above_the_rocm_floor_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _grace(monkeypatch, is_rocm=True, explicit="60", configured=60) == 60.0


def test_rocm_floor_never_shortens_a_larger_generic_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the unset default ever exceeds the ROCm floor, keep the larger one."""
    assert _grace(monkeypatch, is_rocm=True, configured=120) == 120.0
