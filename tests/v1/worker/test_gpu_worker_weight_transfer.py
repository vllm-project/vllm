# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

import pytest

from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.started = False
        self.finished = False
        self.update_calls: list[dict] = []

    def start_weight_update(self) -> None:
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self.finished = True


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    return worker


def test_start_update_finish_delegates_to_engine():
    engine = _RecordingEngine()
    worker = _make_worker(engine)

    Worker.start_weight_update(worker)
    assert engine.started is True
    assert worker._weight_update_active is True

    Worker.update_weights(worker, {"names": ["w"]})
    assert engine.update_calls == [{"names": ["w"]}]
    assert worker._weight_update_active is True

    Worker.finish_weight_update(worker)
    assert engine.finished is True
    assert worker._weight_update_active is False


def test_double_start_raises():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="already"):
        Worker.start_weight_update(worker)


def test_update_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        Worker.update_weights(worker, {"names": ["w"]})


def test_finish_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="without a matching"):
        Worker.finish_weight_update(worker)


def test_update_resets_active_on_error():
    engine = _RecordingEngine(raise_on_update=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    # A failed update ends the session so the next start is clean.
    assert worker._weight_update_active is False


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)
