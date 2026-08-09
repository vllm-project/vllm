# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for how `Worker.determine_available_memory` spends the CUDA graph
memory estimate.

Whether profiling runs at all, and whether its result is subtracted, decides
how much memory the KV cache gets.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.worker import gpu_worker as gw

pytestmark = pytest.mark.cpu_test

GIB = 1 << 30
FREE_MEMORY = 64 * GIB
TOTAL_MEMORY = 80 * GIB
REQUESTED_MEMORY = 60 * GIB
NON_KV_CACHE_MEMORY = 20 * GIB
ESTIMATE = 10 * GIB


class _FakeModelRunner:
    model_memory_usage = 2 * GIB
    cudagraph_profiling_retained_memory = 0

    def __init__(self, estimate: int = ESTIMATE):
        self._estimate = estimate
        self.profile_cudagraph_memory_calls = 0

    def profile_run(self) -> None:
        pass

    def profile_cudagraph_memory(self) -> int:
        self.profile_cudagraph_memory_calls += 1
        return self._estimate


class _RecordingLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning_once(self, msg: str, *args, **kwargs) -> None:
        self.warnings.append(msg)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.warnings.append(msg)

    def info(self, msg: str, *args, **kwargs) -> None:
        pass

    def info_once(self, msg: str, *args, **kwargs) -> None:
        pass

    def debug(self, msg: str, *args, **kwargs) -> None:
        pass


@pytest.fixture
def patched_env(monkeypatch):
    """Neutralize everything `determine_available_memory` reaches for."""
    recording_logger = _RecordingLogger()

    @contextmanager
    def fake_memory_profiling(snapshot, weights_memory):
        yield SimpleNamespace(
            total_consumed=NON_KV_CACHE_MEMORY,
            transient_peak_headroom=1 * GIB,
            non_kv_cache_memory=NON_KV_CACHE_MEMORY,
            after_profile=SimpleNamespace(free_memory=FREE_MEMORY),
        )

    monkeypatch.setattr(gw, "maybe_apply_startup_plan", lambda worker: None)
    monkeypatch.setattr(gw, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(
        gw,
        "reserve_mm_ipc_gpu_memory",
        lambda nbytes, multimodal_config, api_process_count: nbytes,
    )
    monkeypatch.setattr(
        gw, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    monkeypatch.setattr(gw, "logger", recording_logger)
    return SimpleNamespace(monkeypatch=monkeypatch, logger=recording_logger)


def _run(
    patched_env,
    *,
    estimate_cudagraphs: bool,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE,
) -> tuple[int, _FakeModelRunner]:
    patched_env.monkeypatch.setattr(
        gw.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", estimate_cudagraphs
    )
    model_runner = _FakeModelRunner()
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode)
        ),
        model_runner=model_runner,
        init_snapshot=SimpleNamespace(
            free_memory=FREE_MEMORY, total_memory=TOTAL_MEMORY
        ),
        requested_memory=REQUESTED_MEMORY,
    )
    available = gw.Worker.determine_available_memory(worker)
    return available, model_runner


def test_estimate_is_profiled_and_subtracted_from_kv_cache(patched_env):
    available, model_runner = _run(patched_env, estimate_cudagraphs=True)

    assert model_runner.profile_cudagraph_memory_calls == 1
    assert available == REQUESTED_MEMORY - NON_KV_CACHE_MEMORY - ESTIMATE


def test_disabling_the_estimate_skips_profiling_entirely(patched_env):
    """Profiling captures every graph, so computing an estimate that is then
    discarded costs startup time for nothing."""
    available, model_runner = _run(patched_env, estimate_cudagraphs=False)

    assert model_runner.profile_cudagraph_memory_calls == 0
    assert available == REQUESTED_MEMORY - NON_KV_CACHE_MEMORY
    assert any("profiling is disabled" in msg for msg in patched_env.logger.warnings)


def test_no_warning_when_no_graphs_will_be_captured(patched_env):
    """With cudagraph_mode NONE there is no graph memory to account for, so
    warning about unaccounted graph memory would only mislead."""
    _, model_runner = _run(
        patched_env,
        estimate_cudagraphs=False,
        cudagraph_mode=CUDAGraphMode.NONE,
    )

    assert model_runner.profile_cudagraph_memory_calls == 0
    assert patched_env.logger.warnings == []
