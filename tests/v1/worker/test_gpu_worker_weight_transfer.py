# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

import pytest

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.started = False
        self.finished = False
        self.reset_count = 0
        self.update_calls: list[dict] = []
        self.seen_configs: list[VllmConfig] = []

    def _record_config(self) -> None:
        self.seen_configs.append(get_current_vllm_config())

    def start_weight_update(self) -> None:
        self._record_config()
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self._record_config()
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self._record_config()
        self.finished = True

    def reset_weight_update_target(self) -> None:
        self.reset_count += 1


class _RecordingModelRunner:
    """Minimal stand-in for the model runner."""

    def __init__(self) -> None:
        self.seen_config: VllmConfig | None = None
        self.reset_lora_calls = 0

    def reload_weights(self) -> None:
        self.seen_config = get_current_vllm_config()

    def reset_lora_state(self) -> None:
        self.reset_lora_calls += 1


def _make_worker(engine: _RecordingEngine | None, enable_lora: bool = True) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig()
    if enable_lora:
        worker.vllm_config.lora_config = LoRAConfig()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.model_runner = _RecordingModelRunner()
    return worker


def test_reload_weights_sets_current_config():
    worker = _make_worker(None)
    model_runner = _RecordingModelRunner()
    worker.model_runner = model_runner  # type: ignore[assignment]

    Worker.reload_weights(worker)

    assert model_runner.seen_config is worker.vllm_config


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
    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    assert engine.seen_configs == [worker.vllm_config] * 3
    # The transfer engines never go through reload_weights(), so LoRA
    # state (adapter registry, TP>1 logits mapping) must be invalidated
    # when the session finishes.
    assert worker.model_runner.reset_lora_calls == 1


def test_finish_draft_session_keeps_lora_state():
    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    # Bypass draft-model plumbing; only the session bookkeeping matters here.
    worker._set_draft_weight_update_target = lambda: None

    Worker.start_draft_weight_update(worker)
    Worker.finish_weight_update(worker)

    # Draft-model updates don't touch the target model's weights, so its
    # LoRA state stays valid.
    assert worker.model_runner.reset_lora_calls == 0


def test_finish_without_lora_skips_reset():
    worker = _make_worker(_RecordingEngine(), enable_lora=False)

    Worker.start_weight_update(worker)
    Worker.finish_weight_update(worker)

    assert worker.model_runner.reset_lora_calls == 0


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
    assert engine.reset_count == 1
    assert worker._weight_update_active is False


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)
