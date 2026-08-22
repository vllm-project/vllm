#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPUModelRunner (V2) CUDA graph memory profiling.

These exercise the orchestration of ``profile_cudagraph_memory`` on CPU by
building a runner via ``__new__`` and faking the GPU-only helpers, so the
control flow (bootstrap -> capture into a throwaway pool -> teardown) is
covered without a GPU. See https://github.com/vllm-project/vllm/issues/49224.
"""

import contextlib
from types import SimpleNamespace
from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as mrv2

GLOBAL_POOL = "global-pool"
THROWAWAY_POOL = "throwaway-pool"


class _FakeCudaGraphManager:
    def __init__(self, needs_capture: bool) -> None:
        self._needs_capture = needs_capture
        self.pool: Any = GLOBAL_POOL

    def needs_capture(self) -> bool:
        return self._needs_capture


def _make_profiling_runner(
    cudagraph_mode: CUDAGraphMode,
    *,
    needs_capture: bool = True,
    captured_bytes: int = 7 << 30,
) -> Any:
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
    runner.cudagraph_manager = _FakeCudaGraphManager(needs_capture)
    runner.vllm_config = SimpleNamespace()

    events: list[str] = []
    runner.events = events
    runner._init_minimal_kv_cache_for_profiling = lambda: events.append("init")
    runner._teardown_profiling_state = lambda: events.append("teardown")

    def _capture_model() -> int:
        events.append("capture")
        return captured_bytes

    runner.capture_model = _capture_model
    return runner


def _patch_module(monkeypatch) -> None:
    @contextlib.contextmanager
    def _fake_set_current_vllm_config(_cfg):
        yield

    monkeypatch.setattr(
        mrv2, "set_current_vllm_config", _fake_set_current_vllm_config
    )
    monkeypatch.setattr(
        mrv2,
        "current_platform",
        SimpleNamespace(graph_pool_handle=lambda: THROWAWAY_POOL),
    )


def test_profile_cudagraph_memory_disabled_returns_zero(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.NONE)

    result = mrv2.GPUModelRunner.profile_cudagraph_memory(runner)

    assert result == 0
    # No KV-cache bootstrap or teardown when cudagraphs are disabled.
    assert runner.events == []


def test_profile_cudagraph_memory_no_graphs_tears_down(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL, needs_capture=False)

    result = mrv2.GPUModelRunner.profile_cudagraph_memory(runner)

    assert result == 0
    # Bootstrapped then cleaned up, without capturing or touching the pool.
    assert runner.events == ["init", "teardown"]
    assert runner.cudagraph_manager.pool == GLOBAL_POOL


def test_profile_cudagraph_memory_captures_into_throwaway_pool(monkeypatch):
    _patch_module(monkeypatch)
    captured_bytes = 9 << 30
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL, captured_bytes=captured_bytes
    )

    result = mrv2.GPUModelRunner.profile_cudagraph_memory(runner)

    assert result == captured_bytes
    # Bootstrap, capture, and teardown run in order.
    assert runner.events == ["init", "capture", "teardown"]
    # Capture must use a throwaway pool, not the persistent global pool.
    assert runner.cudagraph_manager.pool == THROWAWAY_POOL


def test_profile_cudagraph_memory_tears_down_on_capture_error(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)

    def _boom() -> int:
        runner.events.append("capture")
        raise RuntimeError("capture failed")

    runner.capture_model = _boom

    try:
        mrv2.GPUModelRunner.profile_cudagraph_memory(runner)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected capture error to propagate")

    # Teardown still runs even if capture raises.
    assert runner.events == ["init", "capture", "teardown"]
