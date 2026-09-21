# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.v1.worker.gpu_model_runner import _get_parameter_for_reload
from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    def __init__(
        self,
        raise_on_update: bool = False,
        raise_on_start: bool = False,
        raise_on_finish: bool = False,
        raise_on_abort: bool = False,
    ):
        self.raise_on_update = raise_on_update
        self.raise_on_start = raise_on_start
        self.raise_on_finish = raise_on_finish
        self.raise_on_abort = raise_on_abort
        self.started = False
        self.finished = False
        self.reset_count = 0
        self.abort_count = 0
        self.calls: list[str] = []
        self.supports_draft_weight_update = False
        self.update_calls: list[dict] = []
        self.seen_configs: list[VllmConfig] = []

    def _record_config(self) -> None:
        self.seen_configs.append(get_current_vllm_config())

    def start_weight_update(self) -> None:
        self._record_config()
        self.calls.append("start")
        if self.raise_on_start:
            raise ValueError("boom")
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self._record_config()
        self.calls.append("update")
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self._record_config()
        self.calls.append("finish")
        if self.raise_on_finish:
            raise ValueError("boom")
        self.finished = True

    def abort_weight_update(self) -> None:
        self.calls.append("abort")
        self.abort_count += 1
        if self.raise_on_abort:
            raise RuntimeError("abort failed")

    def reset_weight_update_target(self) -> None:
        self.calls.append("reset")
        self.reset_count += 1


class _RecordingModelRunner:
    def __init__(self) -> None:
        self.seen_config: VllmConfig | None = None
        self.reset_lora_calls = 0

    def reload_weights(self) -> None:
        self.seen_config = get_current_vllm_config()

    def reset_lora_state(self) -> None:
        self.reset_lora_calls += 1


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker._asleep_tags = set()
    worker.model_runner = _RecordingModelRunner()
    return worker


def test_reload_weights_sets_current_config():
    worker = _make_worker(None)
    model_runner = _RecordingModelRunner()
    worker.model_runner = model_runner  # type: ignore[assignment]

    Worker.reload_weights(worker)

    assert model_runner.seen_config is worker.vllm_config


def test_reload_parameter_lookup_preserves_lora_module_names():
    base_layer = nn.Module()
    qweight = nn.Parameter(torch.ones(1))
    base_layer.register_parameter("qweight", qweight)
    wrapper = BaseLayerWithLoRA()
    wrapper.base_layer = base_layer
    model = nn.Module()
    model.proj = wrapper

    named_parameters = dict(model.named_parameters())
    assert set(named_parameters) == {"proj.base_layer.qweight"}
    assert named_parameters["proj.base_layer.qweight"] is qweight
    assert model.get_parameter("proj.base_layer.qweight") is qweight
    assert _get_parameter_for_reload(model, "proj.qweight") is qweight


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
    assert worker.model_runner.reset_lora_calls == 1


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, {"names": ["rank-1"]}), (2, {"names": []})],
)
def test_rank_local_update_selects_worker_payload(rank, expected):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = rank
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker, [{"names": ["rank-0"]}, {"names": ["rank-1"]}, {"names": []}]
    )

    assert engine.update_calls == [expected]
    assert worker._weight_update_active is True


def test_rank_local_update_uses_data_parallel_index_after_reconfigure():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = 0
    parallel_config = ParallelConfig(
        data_parallel_size=4,
        data_parallel_rank=2,
    )
    assert parallel_config.data_parallel_rank == 2
    assert parallel_config.data_parallel_index == 2

    parallel_config.reconfigure_for_independent_dp_rank()
    assert parallel_config.data_parallel_rank == 0
    assert parallel_config.data_parallel_index == 2

    worker.vllm_config.parallel_config = parallel_config
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker,
        [
            {"names": ["dp-0"]},
            {"names": ["dp-1"]},
            {"names": ["dp-2"]},
            {"names": ["dp-3"]},
        ],
    )

    assert engine.update_calls == [{"names": ["dp-2"]}]
    assert worker._weight_update_active is True


def test_finish_draft_session_keeps_lora_state():
    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    worker._set_draft_weight_update_target = lambda: None

    Worker.start_draft_weight_update(worker)
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


def test_update_error_lets_engine_abort_before_target_reset():
    engine = _RecordingEngine(raise_on_update=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    # The engine restores the model while it still points at the session's
    # target, then the target is reset and the session is gone.
    assert engine.calls == ["start", "update", "abort", "reset"]
    assert worker._weight_update_active is False
    with pytest.raises(RuntimeError, match="without a matching"):
        Worker.finish_weight_update(worker)


def test_start_error_lets_engine_abort():
    engine = _RecordingEngine(raise_on_start=True)
    worker = _make_worker(engine)

    with pytest.raises(ValueError, match="boom"):
        Worker.start_weight_update(worker)

    assert engine.calls == ["start", "abort", "reset"]
    assert worker._weight_update_active is False


def test_finish_error_lets_engine_abort_and_drops_session():
    engine = _RecordingEngine(raise_on_finish=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": ["w"]})

    with pytest.raises(ValueError, match="boom"):
        Worker.finish_weight_update(worker)

    assert engine.calls == ["start", "update", "finish", "abort", "reset"]
    assert worker._weight_update_active is False
    assert worker.model_runner.reset_lora_calls == 0


def test_abort_error_does_not_hide_the_update_error():
    engine = _RecordingEngine(raise_on_update=True, raise_on_abort=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    # The target is still reset and the session still ends.
    assert engine.calls == ["start", "update", "abort", "reset"]
    assert worker._weight_update_active is False


def test_successful_session_never_aborts():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": ["w"]})
    Worker.finish_weight_update(worker)

    assert engine.calls == ["start", "update", "finish", "reset"]
    assert engine.abort_count == 0


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)


def test_start_update_rejected_while_weights_asleep():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker._asleep_tags = {"weights", "kv_cache"}
    with pytest.raises(RuntimeError, match="asleep"):
        Worker.start_weight_update(worker)
    assert engine.started is False
    assert worker._weight_update_active is False


def test_update_rejected_while_weights_asleep_resets_session():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)
    # The engine went to sleep between start and the first chunk.
    worker._asleep_tags = {"weights", "kv_cache"}
    with pytest.raises(RuntimeError, match="asleep"):
        Worker.update_weights(worker, {"names": ["w"]})
    assert engine.update_calls == []
    assert worker._weight_update_active is False
    assert engine.reset_count == 1


def test_update_allowed_once_weights_are_resident_again():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker._asleep_tags = {"kv_cache"}  # weights woken, KV cache still asleep
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": ["w"]})
    Worker.finish_weight_update(worker)
    assert engine.update_calls == [{"names": ["w"]}]
    assert engine.finished is True


def test_finish_rejected_while_weights_asleep_resets_session():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": ["w"]})
    # The engine went to sleep after the last chunk and before finish.
    worker._asleep_tags = {"weights", "kv_cache"}
    with pytest.raises(RuntimeError, match="asleep"):
        Worker.finish_weight_update(worker)
    assert engine.finished is False
    assert worker._weight_update_active is False
    assert engine.reset_count == 1


def test_reload_weights_rejected_while_weights_asleep():
    worker = _make_worker(None)
    worker._asleep_tags = {"weights"}
    with pytest.raises(RuntimeError, match="asleep"):
        Worker.reload_weights(worker)
    assert worker.model_runner.seen_config is None


def test_worker_tracks_asleep_tags_across_sleep_and_partial_wake(monkeypatch):
    """Weight updates stay refused until the "weights" tag itself is woken."""
    from types import SimpleNamespace

    class _Backend:
        def suspend(self, level: int = 1) -> None:
            pass

        def resume(self, tags: list[str] | None = None) -> None:
            pass

        def discard(self, tags: tuple[str, ...]) -> None:
            pass

    worker = _make_worker(_RecordingEngine())
    worker._sleep_mode_backend = _Backend()
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_buffers = {}
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_nccl_comm_suspend=False)
    )
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
    monkeypatch.setattr("torch.accelerator.get_memory_info", lambda: (0, 0))
    monkeypatch.setattr(Worker, "synchronize_device", lambda self: None)

    worker._require_resident_weights("update weights")  # awake: allowed

    worker.sleep(level=1)
    assert worker._asleep_tags == {"weights", "kv_cache"}
    with pytest.raises(RuntimeError, match="asleep"):
        worker._require_resident_weights("update weights")

    worker.wake_up(tags=["kv_cache"])
    assert worker._asleep_tags == {"weights"}
    with pytest.raises(RuntimeError, match="asleep"):
        worker._require_resident_weights("update weights")

    worker.wake_up(tags=["weights"])
    assert worker._asleep_tags == set()
    worker._require_resident_weights("update weights")

    worker.sleep(level=1)
    worker.wake_up()
    assert worker._asleep_tags == set()

    worker.discard(("kv_cache",))
    assert worker._asleep_tags == {"kv_cache"}
    worker._require_resident_weights("update weights")  # only the KV cache is gone
