# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    supports_draft_weight_update = True
    supports_lora_weight_update = True

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.started = False
        self.finished = False
        self.reset_count = 0
        self.update_calls: list[dict] = []
        self.target = None
        self.requires_model_reload = True

    def set_weight_update_target(
        self, target, model_config, *, requires_model_reload=True
    ) -> None:
        del model_config
        self.target = target
        self.requires_model_reload = requires_model_reload

    def start_weight_update(self) -> None:
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")
        if self.target is not None and "weights" in update_info:
            self.target.load_weights(update_info["weights"])

    def finish_weight_update(self) -> None:
        self.finished = True

    def reset_weight_update_target(self) -> None:
        self.reset_count += 1
        self.target = None
        self.requires_model_reload = True


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._lora_weight_update_target = None
    worker.model_config = MagicMock()
    return worker


def _attach_lora_manager(worker: Worker, adapter_ids: set[int]):
    lora_manager = MagicMock()
    lora_manager.list_adapters.return_value = adapter_ids
    worker.model_runner = SimpleNamespace(
        lora_manager=lora_manager,
        _ensure_lora_enabled=lambda: None,
    )
    return lora_manager


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

    assert engine.reset_count == 1
    assert worker._weight_update_active is False


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)


def test_lora_weight_update_uses_engine_target_and_replaces_adapter(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    lora_manager = _attach_lora_manager(worker, {1})
    tensor_name = "base_model.model.q_proj.lora_A.weight"

    Worker.start_lora_weight_update(
        worker,
        {
            "lora_int_id": 1,
            "peft_config": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            },
            "tensor_names": [tensor_name],
        },
    )
    assert engine.target is worker._lora_weight_update_target
    assert engine.requires_model_reload is False

    source_tensor = torch.ones(2, 2)
    Worker.update_weights(worker, {"weights": [(tensor_name, source_tensor)]})
    source_tensor.zero_()
    Worker.finish_weight_update(worker)

    call = lora_manager.replace_adapter_from_tensors.call_args
    assert call.kwargs["lora_int_id"] == 1
    assert torch.equal(call.kwargs["tensors"][tensor_name], torch.ones(2, 2))
    assert engine.target is None
    assert worker._weight_update_active is False
    assert worker._lora_weight_update_target is None


def test_lora_weight_update_requires_existing_adapter():
    worker = _make_worker(_RecordingEngine())
    _attach_lora_manager(worker, set())

    with pytest.raises(ValueError, match="must be loaded"):
        Worker.start_lora_weight_update(
            worker,
            {
                "lora_int_id": 1,
                "peft_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj"],
                },
                "tensor_names": ["lora_A"],
            },
        )

    assert worker._weight_update_active is False


def test_lora_weight_update_rejects_unsupported_engine():
    engine = _RecordingEngine()
    engine.supports_lora_weight_update = False
    worker = _make_worker(engine)
    _attach_lora_manager(worker, {1})

    with pytest.raises(RuntimeError, match="does not support LoRA"):
        Worker.start_lora_weight_update(
            worker,
            {
                "lora_int_id": 1,
                "peft_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj"],
                },
                "tensor_names": ["lora_A"],
            },
        )


def test_finish_lora_weight_update_rejects_incomplete_manifest():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    lora_manager = _attach_lora_manager(worker, {1})
    Worker.start_lora_weight_update(
        worker,
        {
            "lora_int_id": 1,
            "peft_config": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            },
            "tensor_names": ["lora_A", "lora_B"],
        },
    )
    Worker.update_weights(worker, {"weights": [("lora_A", torch.ones(2, 2))]})

    with pytest.raises(ValueError, match="manifest mismatch"):
        Worker.finish_weight_update(worker)

    lora_manager.replace_adapter_from_tensors.assert_not_called()
    assert engine.target is None
    assert worker._weight_update_active is False
    assert worker._lora_weight_update_target is None
