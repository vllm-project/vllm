# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm mono lifecycle guards without allocating model or IPC storage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.models.minimax_m3.amd import mono, mono_worker
from vllm.models.minimax_m3.amd.mono_worker import M3MonoModelRunner, M3MonoWorker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


def test_weight_transfer_is_rejected_before_loading():
    runner = M3MonoModelRunner.__new__(M3MonoModelRunner)
    runner.vllm_config = SimpleNamespace(weight_transfer_config=object())
    with pytest.raises(ValueError, match="does not support weight transfer"):
        runner.load_model()


def test_weight_reload_is_rejected_before_mutation():
    runner = M3MonoModelRunner.__new__(M3MonoModelRunner)
    with pytest.raises(ValueError, match="does not support weight reloading"):
        runner.reload_weights()


def test_shared_runner_ignores_model_specific_flag(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

    monkeypatch.setenv("VLLM_ROCM_USE_ATOM_M3_MONO", "1")
    reload_weights = Mock()
    monkeypatch.setattr(GPUModelRunnerV1, "reload_weights", reload_weights)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.reload_weights()
    reload_weights.assert_called_once_with(runner)


@pytest.mark.parametrize("enabled", [False, True])
def test_weight_reset_preserves_converted_copies(monkeypatch, enabled):
    monkeypatch.setenv("VLLM_ROCM_USE_ATOM_M3_MONO", str(int(enabled)))
    reset = Mock()
    monkeypatch.setattr(Worker, "reset_weights", reset)
    worker = M3MonoWorker.__new__(M3MonoWorker)
    if enabled:
        with pytest.raises(ValueError, match="does not support weight reset"):
            worker.reset_weights()
        reset.assert_not_called()
    else:
        worker.reset_weights()
        reset.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_selects_native_runner_when_disabled(monkeypatch, enabled):
    monkeypatch.setenv("VLLM_ROCM_USE_ATOM_M3_MONO", str(int(enabled)))
    native, specialized = object(), object()
    monkeypatch.setattr(Worker, "_make_model_runner", lambda self: native)
    monkeypatch.setattr(mono_worker, "M3MonoModelRunner", lambda *args: specialized)
    worker = M3MonoWorker.__new__(M3MonoWorker)
    worker.use_v2_model_runner = True
    worker.vllm_config = SimpleNamespace(is_mm_encoder_only=False)
    worker.device = None
    assert worker._make_model_runner() is (specialized if enabled else native)


@pytest.mark.parametrize("profiling", [False, True])
def test_library_binds_final_native_cache_only(monkeypatch, profiling):
    runner = M3MonoModelRunner.__new__(M3MonoModelRunner)
    runner.model, runner.vllm_config = object(), object()
    allocated_cache = object()

    def allocate(self, *args):
        self.kv_cache_config = allocated_cache

    monkeypatch.setattr(GPUModelRunner, "initialize_kv_cache", allocate)
    prepare = Mock()
    monkeypatch.setattr(mono, "prepare_model", prepare)
    runner.initialize_kv_cache(object(), is_profiling=profiling)
    if profiling:
        prepare.assert_not_called()
    else:
        prepare.assert_called_once_with(
            runner.model, runner.vllm_config, allocated_cache
        )


@pytest.mark.parametrize("graph_cleanup_fails", [False, True])
def test_ipc_survives_until_graph_teardown_finishes(monkeypatch, graph_cleanup_fails):
    runner = M3MonoModelRunner.__new__(M3MonoModelRunner)
    model = runner.model = object()
    graph_alive = True

    def shutdown(self):
        nonlocal graph_alive
        if graph_cleanup_fails:
            raise RuntimeError("graph cleanup failed")
        graph_alive = False
        del self.model

    def release(retained_model):
        assert not graph_alive
        assert retained_model is model

    monkeypatch.setattr(GPUModelRunner, "shutdown", shutdown)
    close = Mock(side_effect=release)
    monkeypatch.setattr(mono, "release_model", close)
    monkeypatch.setattr(mono_worker.gc, "collect", lambda: None)
    flush = Mock(side_effect=lambda: close.assert_called_once_with(model))
    monkeypatch.setattr(mono_worker.torch.accelerator, "empty_cache", flush)
    if graph_cleanup_fails:
        with pytest.raises(RuntimeError, match="graph cleanup failed"):
            runner.shutdown()
        close.assert_not_called()
        flush.assert_not_called()
    else:
        runner.shutdown()
        close.assert_called_once_with(model)
        flush.assert_called_once()
