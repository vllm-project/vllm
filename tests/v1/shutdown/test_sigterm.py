# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SIGTERM handling while EngineCore and worker processes shut down."""

import multiprocessing
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from vllm.v1.executor.multiproc_executor import _make_worker_signal_handler


def test_worker_acts_on_first_sigterm_after_shutdown_request() -> None:
    """The death-pipe monitor sets shutdown_requested but cannot wake a worker
    blocked on a remote MQ read, so the first SIGTERM must still exit it."""
    shutdown_requested, teardown_started = threading.Event(), threading.Event()
    handler = _make_worker_signal_handler(shutdown_requested, teardown_started)
    shutdown_requested.set()

    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
    handler(signal.SIGTERM, None)  # a second signal is ignored


def test_worker_ignores_sigterm_during_teardown() -> None:
    shutdown_requested, teardown_started = threading.Event(), threading.Event()
    handler = _make_worker_signal_handler(shutdown_requested, teardown_started)
    teardown_started.set()

    handler(signal.SIGTERM, None)
    assert not shutdown_requested.is_set()


class _EngineSignalledDuringTeardown:
    """Stands in for EngineCoreProc, which needs a model: the busy loop exits
    at once and teardown receives the process manager's follow-up SIGTERM."""

    marker: Path

    def __init__(self, *args, **kwargs) -> None:
        self.shutdown_state = None

    def run_busy_loop(self) -> None:
        raise SystemExit(0)

    def shutdown(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.5)
        self.marker.write_text("teardown finished")


def _run_engine_core(marker: Path) -> None:
    from vllm.config import VllmConfig
    from vllm.v1.engine import core

    _EngineSignalledDuringTeardown.marker = marker
    run_engine_core = core.EngineCoreProc.run_engine_core
    core.EngineCoreProc = _EngineSignalledDuringTeardown  # type: ignore[misc]
    run_engine_core(vllm_config=VllmConfig())


def test_engine_core_teardown_survives_sigterm(tmp_path: Path) -> None:
    marker = tmp_path / "teardown"
    proc = multiprocessing.get_context("spawn").Process(
        target=_run_engine_core, args=(marker,)
    )
    proc.start()
    proc.join(120)

    assert marker.exists(), f"teardown interrupted, exitcode={proc.exitcode}"
    assert proc.exitcode == 0
