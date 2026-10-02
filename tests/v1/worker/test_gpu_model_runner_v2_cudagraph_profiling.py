#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPUModelRunner (V2) CUDA graph memory profiling.

These exercise the orchestration of ``profile_cudagraph_memory`` on CPU by
building a runner via ``__new__`` and faking the GPU-only helpers, so the
control flow (bootstrap -> sample FULL graphs into a throwaway pool ->
extrapolate -> teardown) is covered without a GPU.
See https://github.com/vllm-project/vllm/issues/49224.
"""

import contextlib
import gc
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu import model_runner as mrv2

GLOBAL_POOL = "global-pool"
THROWAWAY_POOL = "throwaway-pool"


class _FakeCudaGraphManager:
    def __init__(
        self, needs_capture: bool, num_full_descs: int, piecewise_only: bool = False
    ) -> None:
        self._needs_capture = needs_capture
        self.pool: Any = GLOBAL_POOL
        descs = [object() for _ in range(num_full_descs)]
        if piecewise_only:
            self._capture_descs = {CUDAGraphMode.PIECEWISE: descs}
        else:
            self._capture_descs = {CUDAGraphMode.FULL: descs} if needs_capture else {}
        # Profiling hooks set by profile_cudagraph_memory.
        self._max_full_descs_to_capture: int | None = None
        self._capture_mem_samples: list[int] | None = None
        self.use_breakable_cg = False

    def needs_capture(self) -> bool:
        return self._needs_capture


def _make_profiling_runner(
    cudagraph_mode: CUDAGraphMode,
    *,
    needs_capture: bool = True,
    num_full_descs: int = 3,
    piecewise_only: bool = False,
    captured_bytes: int = 7 << 30,
    mem_samples: list[int] | None = None,
) -> Any:
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
    runner.cudagraph_manager = _FakeCudaGraphManager(
        needs_capture, num_full_descs, piecewise_only
    )
    runner.vllm_config = SimpleNamespace()

    events: list[str] = []
    runner.events = events

    def _capture_model(*, profile_only: bool = False) -> int:
        assert profile_only
        events.append("capture")
        # Simulate the manager's per-FULL-graph memory sampling.
        samples = runner.cudagraph_manager._capture_mem_samples
        if samples is not None:
            samples.extend(mem_samples or [])
        return captured_bytes

    runner.capture_model = _capture_model
    return runner


class _FakePlatform:
    """Stands in for current_platform; the global graph pool is a class attr,
    matching vllm.platforms.Platform's lazy singleton."""

    _global_graph_pool: Any = GLOBAL_POOL

    @staticmethod
    def graph_pool_handle() -> Any:
        return THROWAWAY_POOL

    def get_global_graph_pool(self) -> Any:
        return type(self)._global_graph_pool


def _patch_module(monkeypatch) -> None:
    @contextlib.contextmanager
    def _fake_set_current_vllm_config(_cfg):
        yield

    _FakePlatform._global_graph_pool = GLOBAL_POOL
    monkeypatch.setattr(cgu, "set_current_vllm_config", _fake_set_current_vllm_config)
    monkeypatch.setattr(cgu, "current_platform", _FakePlatform())
    monkeypatch.setattr(
        cgu, "_init_minimal_kv_cache_for_profiling", lambda r: r.events.append("init")
    )
    monkeypatch.setattr(
        cgu, "_teardown_profiling_state", lambda r: r.events.append("teardown")
    )
    # The profiler reads free GPU memory before/after to compute what it
    # retained; default to a constant (nothing retained).
    monkeypatch.setattr(cgu.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        cgu.torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )


def test_profile_cudagraph_memory_disabled_returns_zero(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.NONE)

    result = cgu.profile_cudagraph_memory(runner)

    assert result == 0
    # No KV-cache bootstrap or teardown when cudagraphs are disabled.
    assert runner.events == []


def test_profile_cudagraph_memory_no_graphs_tears_down(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL, needs_capture=False)

    result = cgu.profile_cudagraph_memory(runner)

    assert result == 0
    # Bootstrapped then cleaned up, without capturing or touching the pool.
    assert runner.events == ["init", "teardown"]
    assert runner.cudagraph_manager.pool == GLOBAL_POOL


def test_profile_cudagraph_memory_samples_and_extrapolates(monkeypatch):
    _patch_module(monkeypatch)
    gib = 1 << 30
    # Measured delta 1000 MiB includes the sampled FULL graphs (100 + 20 MiB).
    # Extrapolated FULL cost for 3 graphs: 100 + 2 * 20 = 140 MiB.
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL,
        num_full_descs=3,
        captured_bytes=1000 * gib,
        mem_samples=[100 * gib, 20 * gib],
    )

    result = cgu.profile_cudagraph_memory(runner)

    assert result == (1000 - (100 + 20) + (100 + 2 * 20)) * gib
    # Bootstrap, capture, and teardown run in order.
    assert runner.events == ["init", "capture", "teardown"]
    # Capture must use a throwaway pool, not the persistent global pool.
    assert runner.cudagraph_manager.pool == THROWAWAY_POOL
    # FULL capture must be limited to the largest few graphs.
    assert (
        runner.cudagraph_manager._max_full_descs_to_capture
        == cgu._FULL_GRAPH_PROFILING_SAMPLES
    )


def test_profile_cudagraph_memory_piecewise_only_returns_measured(monkeypatch):
    _patch_module(monkeypatch)
    captured_bytes = 5 << 30
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        piecewise_only=True,
        captured_bytes=captured_bytes,
    )

    result = cgu.profile_cudagraph_memory(runner)

    # No FULL graphs to sample or extrapolate: the measured delta is exact.
    assert result == captured_bytes


def test_profile_cudagraph_memory_tears_down_on_capture_error(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)

    def _boom(*, profile_only: bool = False) -> int:
        runner.events.append("capture")
        raise RuntimeError("capture failed")

    runner.capture_model = _boom

    try:
        cgu.profile_cudagraph_memory(runner)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected capture error to propagate")

    # Teardown still runs even if capture raises.
    assert runner.events == ["init", "capture", "teardown"]


def test_profile_cudagraph_memory_restores_compilation_counters(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)

    def _capture_model(*, profile_only: bool = False) -> int:
        compilation_counter.num_cudagraph_captured += 5
        compilation_counter.num_gpu_runner_capture_triggers += 1
        return 1 << 30

    runner.capture_model = _capture_model
    captured_before = compilation_counter.num_cudagraph_captured
    triggers_before = compilation_counter.num_gpu_runner_capture_triggers

    cgu.profile_cudagraph_memory(runner)

    # Profiling captures are discarded, so they must not inflate the
    # compilation counters; the real capture_model() runs later.
    assert compilation_counter.num_cudagraph_captured == captured_before
    assert compilation_counter.num_gpu_runner_capture_triggers == triggers_before


def test_model_runner_delegates_to_cudagraph_utils(monkeypatch):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    monkeypatch.setattr(mrv2, "_profile_cudagraph_memory", lambda r: 42)
    assert runner.profile_cudagraph_memory() == 42


def test_extrapolate_full_graph_memory():
    mib = 1 << 20
    # No samples (e.g. no FULL graphs): nothing to add.
    assert cgu._extrapolate_full_graph_memory([], 0) == 0
    # A single graph costs exactly its sample.
    assert cgu._extrapolate_full_graph_memory([100 * mib], 1) == 100 * mib
    # First capture + per-graph cost for the rest.
    assert (
        cgu._extrapolate_full_graph_memory([100 * mib, 20 * mib], 5)
        == (100 + 4 * 20) * mib
    )
    # Per-graph cost is floored to account for driver overhead.
    assert cgu._extrapolate_full_graph_memory([100 * mib, 0], 3) == (100 + 2 * 1) * mib


def test_profile_cudagraph_memory_clears_captured_graphs(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    cleared: list[str] = []
    monkeypatch.setattr(
        cgu.CUDAGraphWrapper,
        "clear_all_graphs",
        classmethod(lambda cls: cleared.append("piecewise")),
    )
    monkeypatch.setattr(
        cgu.BreakableCUDAGraphWrapper,
        "clear_all_graphs",
        classmethod(lambda cls: cleared.append("breakable")),
    )

    cgu.profile_cudagraph_memory(runner)

    # Profiling captures are discarded so the real capture re-captures them
    # against the KV cache.
    assert cleared == ["piecewise", "breakable"]


def test_profile_cudagraph_memory_redirects_wrapper_pools(monkeypatch):
    """Piecewise wrappers must capture into the throwaway pool too.

    Profiling graphs captured into the persistent global pool and then
    discarded drop the pool's use_count to 0, tripping the c10 allocator's
    create_or_incref_pool assert when the real capture reuses that pool.
    """
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    class _FakeWrapper:
        def __init__(self) -> None:
            self.graph_pool: Any = GLOBAL_POOL
            self.pool_during_capture: Any = None

        def clear_graphs(self) -> None:
            pass

    wrapper = _FakeWrapper()
    cgu.CUDAGraphWrapper._all_instances.add(wrapper)
    try:
        capture_model = runner.capture_model

        def _capture_model(*, profile_only: bool = False) -> int:
            wrapper.pool_during_capture = wrapper.graph_pool
            return capture_model(profile_only=profile_only)

        runner.capture_model = _capture_model

        cgu.profile_cudagraph_memory(runner)

        assert wrapper.pool_during_capture == THROWAWAY_POOL
        assert wrapper.graph_pool == GLOBAL_POOL
    finally:
        cgu.CUDAGraphWrapper._all_instances.discard(wrapper)


def test_profile_cudagraph_memory_swaps_and_drops_speculator_managers(monkeypatch):
    """Speculator cudagraph managers must also capture into the throwaway pool.

    They are created during the profiling KV-cache bootstrap, binding the
    (swapped) global graph pool, and are re-created by the real
    initialize_kv_cache; profiling-captured graphs must be dropped at
    teardown rather than released against the persistent global pool, which
    would trip the c10 create_or_incref_pool assert at the real capture.
    """
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    def _init(r):
        r.events.append("init")
        # Mirror production: the speculator's cudagraph managers are created
        # during the profiling KV-cache bootstrap and bind the global pool
        # (which profiling has already pointed at the throwaway pool).
        manager = cgu.CudaGraphManager.__new__(cgu.CudaGraphManager)
        manager.pool = cgu.current_platform.get_global_graph_pool()
        r.speculator = SimpleNamespace(cudagraph_manager=manager)

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", _init)

    pools_seen: list[Any] = []
    capture_model = runner.capture_model

    def _capture_model(*, profile_only: bool = False) -> int:
        pools_seen.append(runner.speculator.cudagraph_manager.pool)
        return capture_model(profile_only=profile_only)

    runner.capture_model = _capture_model

    cgu.profile_cudagraph_memory(runner)

    assert pools_seen == [THROWAWAY_POOL]
    assert runner.speculator.cudagraph_manager is None
    # The real global pool is restored afterwards.
    assert _FakePlatform._global_graph_pool == GLOBAL_POOL


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not cgu.current_platform.is_cuda(), reason="requires CUDA")
def test_profile_cudagraph_memory_frees_throwaway_pool(monkeypatch):
    """Profiling graph memory must be freed before teardown completes."""

    @contextlib.contextmanager
    def _fake_set_current_vllm_config(_cfg):
        yield

    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.compilation_config.static_forward_context = {}
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.kv_caches = []
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace()
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None
    allocation_bytes = 64 << 20
    memory: dict[str, int] = {}

    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    memory["before"] = torch.accelerator.memory_reserved()

    def _init(r):
        r.events.append("init")
        kv_cache = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda")
        r.kv_caches = [kv_cache]
        r.compilation_config.static_forward_context = {
            "layer": SimpleNamespace(kv_cache=kv_cache)
        }
        manager = cgu.CudaGraphManager.__new__(cgu.CudaGraphManager)
        manager.pool = cgu.current_platform.get_global_graph_pool()
        r.speculator = SimpleNamespace(cudagraph_manager=manager)

    def _capture_model(*, profile_only: bool = False) -> int:
        for owner in (
            runner.cudagraph_manager,
            runner.speculator.cudagraph_manager,
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=owner.pool):
                output = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda")
                output.fill_(1)
            owner.profiling_resources = (graph, output)
        torch.accelerator.synchronize()
        memory["captured"] = torch.accelerator.memory_reserved()
        return memory["captured"] - memory["before"]

    monkeypatch.setattr(cgu, "set_current_vllm_config", _fake_set_current_vllm_config)
    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", _init)
    runner.capture_model = _capture_model

    cgu.profile_cudagraph_memory(runner)
    memory["after"] = torch.accelerator.memory_reserved()

    assert memory["captured"] - memory["before"] >= 3 * allocation_bytes
    assert memory["after"] == memory["before"]


def test_teardown_profiling_state_clears_mamba_align_metadata(monkeypatch):
    """Profiling-cached Mamba align metadata must be invalidated at teardown.

    ``MambaHybridModelState`` lazily caches ``_mamba_group_ids`` and
    ``_mamba_spec`` from whichever KVCacheConfig it first sees. When the
    profiling config's group layout differs from the real (e.g. PP-projected)
    config, reusing the stale metadata mismatches the real block tables
    ("expected 3 block tables, got 4" at
    ``MambaSpecDecodeGPUContext.initialize_from_forward_context``).
    """
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False,
        _mamba_ctx=object(),
        _mamba_group_ids=[0, 1],
        _mamba_spec=object(),
    )
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.kv_caches = []
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace()
    runner.cudagraph_manager = object()
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None

    monkeypatch.setattr(cgu.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(cgu.torch.accelerator, "empty_cache", lambda: None)

    cgu._teardown_profiling_state(runner)

    assert runner.model_state._mamba_ctx is None
    assert runner.model_state._mamba_group_ids == []
    assert runner.model_state._mamba_spec is None
