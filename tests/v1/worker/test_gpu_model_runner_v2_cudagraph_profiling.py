#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPUModelRunner (V2) CUDA graph memory profiling.

These exercise the orchestration of `profile_cudagraph_memory` on CPU by
building a runner via `__new__` and faking the GPU-only helpers, so the
control flow (bootstrap -> sample FULL graphs into a throwaway pool ->
extrapolate -> teardown) is covered without a GPU.
See https://github.com/vllm-project/vllm/issues/49224.
"""

import contextlib
import gc
import multiprocessing
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.utils import call_in_spawned_process, create_new_process_for_each_test
from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.eplb_utils import EPLBController
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.spec_decode.standalone_ar.config import (
    StandaloneARSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
    StandaloneARSpeculator,
)

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
    encoder_only: bool = False,
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
    runner.speculator = None
    runner.vllm_config = SimpleNamespace()
    runner.model_config = SimpleNamespace(enforce_eager=False)
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=encoder_only,
        encoder_runner=SimpleNamespace(has_cudagraph=lambda: encoder_only),
        needs_cudagraph_capture=lambda: False,
    )

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
    runner = _make_profiling_runner(CUDAGraphMode.NONE, needs_capture=False)
    runner.cudagraph_manager = None

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


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.NONE, CUDAGraphMode.FULL]
)
def test_profile_cudagraph_memory_piecewise_or_encoder_returns_measured(
    monkeypatch, mode
):
    _patch_module(monkeypatch)
    captured_bytes = 5 << 30
    runner = _make_profiling_runner(
        mode,
        needs_capture=mode == CUDAGraphMode.FULL_AND_PIECEWISE,
        encoder_only=mode != CUDAGraphMode.FULL_AND_PIECEWISE,
        piecewise_only=mode == CUDAGraphMode.FULL_AND_PIECEWISE,
        captured_bytes=captured_bytes,
    )
    manager = runner.cudagraph_manager
    runner.cudagraph_manager = None

    def init(r):
        r.events.append("init")
        r.cudagraph_manager = manager

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", init)

    result = cgu.profile_cudagraph_memory(runner)

    # No FULL graphs to sample or extrapolate: the measured delta is exact.
    assert result == captured_bytes
    assert runner.events == ["init", "capture", "teardown"]
    assert _FakePlatform._global_graph_pool == GLOBAL_POOL


@pytest.mark.parametrize("encoder_only", [False, True])
def test_profile_cudagraph_memory_tears_down_on_capture_error(
    monkeypatch, encoder_only
):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(
        CUDAGraphMode.NONE if encoder_only else CUDAGraphMode.FULL,
        needs_capture=not encoder_only,
        encoder_only=encoder_only,
    )

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
    assert _FakePlatform._global_graph_pool == GLOBAL_POOL


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


@pytest.mark.parametrize("extensible", [False, True])
def test_model_runner_delegates_to_cudagraph_utils(monkeypatch, extensible):
    """The extensible KV cache measures the capture footprint after warmup
    instead of estimating it."""
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.cache_config = SimpleNamespace(enable_extensible_kv_cache=extensible)
    runner.eplb = EPLBController(
        SimpleNamespace(enable_eplb=False), torch.device("cpu")
    )
    monkeypatch.setattr(mrv2, "_profile_cudagraph_memory", lambda r: 42)
    assert runner.profile_cudagraph_memory() == (0 if extensible else 42)


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
@pytest.mark.skipif(
    not cgu.current_platform.is_cuda_alike(), reason="requires CUDA or ROCm"
)
def test_profile_cudagraph_memory_frees_throwaway_pool(monkeypatch):
    """Profiling graph memory must be freed before teardown completes."""

    @contextlib.contextmanager
    def _fake_set_current_vllm_config(_cfg):
        yield

    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.compilation_config.static_forward_context = {}
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, needs_cudagraph_capture=lambda: False
    )
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
    """The Mamba align context built for the profiling KVCacheConfig must be
    dropped at teardown; the real (e.g. PP-projected) config may place Mamba
    layers into a different group layout."""
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False,
        _mamba_ctx=object(),
        needs_cudagraph_capture=lambda: False,
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


# The model-state graph hooks for a model without persistent state, through
# the real estimate and the real warmup. A scenario is a model-state class and
# engine arguments for a real model with dummy weights: facebook/opt-125m
# under the default `cudagraph_mode`, or SmolVLM-256M for encoder graphs,
# with the extensible KV cache off so that the estimate runs. `_observe_opt`
# starts it once, in a new process, with the real constructors, and returns
# plain data that the tests below assert on.

MiB = 1 << 20
START_TIMEOUT = 900

# (hook, info) per recorded call, in call order.
_HOOK_CALLS: list[tuple[str, dict]] = []
# The exceptions the recording model states raised, in order.
_RAISED: list[BaseException] = []
# The recording workers built in this process.
_WORKERS: list[Any] = []
# (profile_only, bytes) for each `capture_model` call of a recording runner.
_CAPTURE_RESULTS: list[tuple[bool, int]] = []


class _HookError(RuntimeError):
    """What a recording model state raises when asked to fail."""


def _record(hook: str, **info: Any) -> dict:
    _HOOK_CALLS.append((hook, info))
    return info


class _RecordingModelState(DefaultModelState):
    """Records the hook calls. It keeps the default capture predicate."""

    captures = False
    fails: frozenset[str] = frozenset()

    def _maybe_fail(self, hook: str) -> None:
        if hook in self.fails:
            error = _HookError(f"{hook} failed")
            _RAISED.append(error)
            raise error

    def initialize_kv_cache(self, kv_cache_config, block_tables) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        _record("initialize_kv_cache")

    def capture_inner_cudagraphs(self, *args, **kwargs) -> None:
        super().capture_inner_cudagraphs(*args, **kwargs)
        _record("capture_inner_cudagraphs")
        self._maybe_fail("capture_inner_cudagraphs")

    def needs_cudagraph_capture(self) -> bool:
        _record("needs_cudagraph_capture")
        return self.captures

    def capture_cudagraphs(self, *, graph_pool, stream, profile_only) -> None:
        runner = _WORKERS[-1].model_runner
        lora_manager = getattr(runner, "lora_manager", None)
        info = _record(
            "capture_cudagraphs",
            profile_only=profile_only,
            has_bindings=self.state_slots is not None,
            graphs_captured=compilation_counter.num_cudagraph_captured,
            stream_is_current=torch.cuda.current_stream(self.device) == stream,
            dummy_loras=(
                None if lora_manager is None else sorted(lora_manager.list_adapters())
            ),
            encoder_graphs_captured=(
                self.supports_mm_inputs
                and self.encoder_runner.cudagraph_manager is not None
                and self.encoder_runner.cudagraph_manager.is_captured()
            ),
        )
        # A fresh allocation: with the caching allocator emptied, the 64 MiB
        # cannot come out of memory that an earlier capture left reserved.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(self.device)[0]
        reserved = torch.cuda.memory_reserved(self.device)
        self._scratch = torch.zeros(64 * MiB, dtype=torch.uint8, device=self.device)
        torch.cuda.synchronize(self.device)
        info["device_bytes_taken"] = free - torch.cuda.mem_get_info(self.device)[0]
        info["reserved_bytes_taken"] = (
            torch.cuda.memory_reserved(self.device) - reserved
        )

    def clear_cudagraphs(self) -> None:
        _record("clear_cudagraphs", has_bindings=self.state_slots is not None)
        self._scratch = None
        self._maybe_fail("clear_cudagraphs")

    def dummy_sampler_run(self, sampler, hidden_states) -> bool:
        _record("dummy_sampler_run")
        return super().dummy_sampler_run(sampler, hidden_states)

    def post_warmup(self) -> None:
        from vllm.v1.worker.workspace import current_workspace_manager

        _record("post_warmup", workspace_locked=current_workspace_manager().is_locked())

    def shutdown(self) -> None:
        _record("shutdown")


class _CapturingModelState(_RecordingModelState):
    """A model state that captures graphs of its own."""

    captures = True


class _InnerCaptureFailsModelState(_CapturingModelState):
    """Its inner graphs, captured with the decoder's, fail."""

    fails = frozenset({"capture_inner_cudagraphs"})


class _InnerCaptureAndReleaseFailModelState(_CapturingModelState):
    fails = frozenset({"capture_inner_cudagraphs", "clear_cudagraphs"})


class _RecordingRunner(mrv2.GPUModelRunner):
    """The standard runner; it records what each `capture_model` returns."""

    def capture_model(self, *, profile_only: bool = False) -> int:
        captured = super().capture_model(profile_only=profile_only)
        _CAPTURE_RESULTS.append((profile_only, int(captured)))
        return captured


class _RecordingWorker(gpu_worker.Worker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _WORKERS.append(self)

    def _make_model_runner(self) -> mrv2.GPUModelRunner:
        return _RecordingRunner(self.vllm_config, self.device)


class _RecordingSpeculator(StandaloneARSpeculator):
    """The draft-model speculator; it records when its capture is done."""

    def capture(self) -> None:
        super().capture()
        _record("speculator_capture")


class _RecordingSpeculatorConfig(StandaloneARSpeculatorConfig):
    speculator_class = f"{__name__}.{_RecordingSpeculator.__qualname__}"


class _OwnCaptureRunner(mrv2.GPUModelRunner):
    """Replaces capture and its predicate without calling `GPUModelRunner`'s,
    as the encoder-only runner does."""

    def needs_cudagraph_capture(self) -> bool:
        return False

    def capture_model(self, *, profile_only: bool = False) -> int:
        return 0


class _OwnCaptureRunnerWorker(_RecordingWorker):
    def _make_model_runner(self) -> mrv2.GPUModelRunner:
        return _OwnCaptureRunner(self.vllm_config, self.device)


_OPT = dict(
    model="facebook/opt-125m",
    base="vllm.model_executor.models.opt.OPTForCausalLM",
    max_model_len=64,
)
# A small model whose vision encoder supports CUDA graphs. `cudagraph_mode`
# is NONE, so the encoder graphs are the runner's only ones.
_SMOLVLM = dict(
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    base="vllm.model_executor.models.idefics3.Idefics3ForConditionalGeneration",
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1},
    compilation_config={
        "cudagraph_mode": "NONE",
        "cudagraph_mm_encoder": True,
        "encoder_cudagraph_token_budgets": [4096],
    },
)


def _opt(state: type[DefaultModelState], *, start: str = "estimate", **engine_args):
    return dict(state=state, start=start, engine_args={**_OPT, **engine_args})


_OPT_SCENARIOS = {
    "default-predicate": _opt(_RecordingModelState),
    "own-capture-runner": _opt(
        _CapturingModelState,
        worker_cls=f"{__name__}.{_OwnCaptureRunnerWorker.__qualname__}",
    ),
    "capturing": _opt(_CapturingModelState),
    "capturing-lora": _opt(
        _CapturingModelState, enable_lora=True, max_loras=1, max_lora_rank=8
    ),
    # A draft model that speculates for the target, through a registered
    # speculative method whose speculator records its capture.
    "capturing-speculator": _opt(
        _CapturingModelState,
        speculative_config={
            "method": "recording_draft_model",
            "model": "facebook/opt-125m",
            "num_speculative_tokens": 2,
        },
    ),
    "inner-capture-fails": _opt(_InnerCaptureFailsModelState),
    "inner-capture-and-release-fail": _opt(_InnerCaptureAndReleaseFailModelState),
    "capturing-start": _opt(_CapturingModelState, start="engine"),
    "pooling-start": _opt(
        _RecordingModelState, start="engine", runner="pooling", convert="embed"
    ),
    "encoder-default-predicate": _opt(_RecordingModelState, **_SMOLVLM),
    "encoder-capturing": _opt(_CapturingModelState, **_SMOLVLM),
}


def _observe_opt(name: str) -> dict:
    """Start scenario `name` in this process and return what was observed."""
    import logging

    from vllm import ModelRegistry
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )

    scenario = _OPT_SCENARIOS[name]
    model_state_cls = scenario["state"]
    engine_args = {
        "worker_cls": f"{__name__}.{_RecordingWorker.__qualname__}",
        **scenario["engine_args"],
    }

    class RecordingModel(resolve_obj_by_qualname(engine_args.pop("base"))):  # type: ignore[misc]
        @classmethod
        def get_model_state_cls(cls) -> type[DefaultModelState]:
            return model_state_cls

    ModelRegistry.register_model("RecordingModel", RecordingModel)
    register_speculative_method("recording_draft_model", _RecordingSpeculatorConfig)
    config = EngineArgs(
        hf_overrides={"architectures": ["RecordingModel"]},
        load_format="dummy",
        max_num_seqs=2,
        enable_extensible_kv_cache=False,
        **engine_args,
    ).create_engine_config()
    seen: dict = dict(
        error=None,
        cudagraph_mode=config.compilation_config.cudagraph_mode.name,
        extensible_kv_cache=config.cache_config.enable_extensible_kv_cache,
    )

    def failure(error: BaseException) -> dict:
        raised = [i for i, e in enumerate(_RAISED) if e is error]
        return dict(type=type(error).__name__, raised=raised[0] if raised else None)

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logging.getLogger("vllm").addHandler(handler)

    if scenario["start"] == "engine":
        core = EngineCore(
            vllm_config=config, executor_class=UniProcExecutor, log_stats=False
        )
        seen["startup_calls"] = len(_HOOK_CALLS)
        core.shutdown()
        seen["calls"] = list(_HOOK_CALLS)
    else:
        executor = UniProcExecutor(config)
        try:
            # What the engine core does before it profiles memory.
            layer_configs = executor.get_kv_cache_layer_configs()
            layout = resolve_kv_cache_layout(
                config,
                executor.get_supported_kv_cache_layouts(),
                [lc for layers in layer_configs for lc in layers.values()],
            )
            groups = get_global_kv_cache_groups(config, layer_configs)
            config.adjust_dcp_kv_cache_interleave_size(groups)
            executor.set_kv_cache_layout(
                layout.name, config.parallel_config.cp_kv_cache_interleave_size
            )
            _HOOK_CALLS.clear()
            try:
                executor.determine_available_memory()
            except Exception as error:
                seen["error"] = failure(error)
            worker = executor.driver_worker.worker
            runner = worker.model_runner
            lora_manager = getattr(runner, "lora_manager", None)
            seen["after_estimate"] = dict(
                estimate=getattr(worker, "cudagraph_memory_estimate", None),
                own_capture_runner=isinstance(runner, _OwnCaptureRunner),
                runner_needs_capture=runner.needs_cudagraph_capture(),
                runner_has_bindings=runner.state_slots is not None,
                loras=(
                    None
                    if lora_manager is None
                    else sorted(lora_manager.list_adapters())
                ),
            )
            # The estimate's calls only: the shutdown below has its own hook.
            seen["calls"] = list(_HOOK_CALLS)
        finally:
            executor.shutdown()

    seen["capture_results"] = list(_CAPTURE_RESULTS)
    seen["logged_raised"] = sorted(
        {
            index
            for record in records
            if record.exc_info and record.exc_info[2] is not None
            for index, error in enumerate(_RAISED)
            if record.exc_info[1] is error
        }
    )
    return seen


@pytest.fixture(scope="module")
def opt_started():
    """Returns what `_observe_opt` saw of a scenario. Each scenario starts
    once per module, in a new process; a start that failed or timed out is
    not repeated, and every test of its scenario fails with its error."""
    observed: dict[str, dict | RuntimeError] = {}

    def start(name: str) -> dict:
        if name not in observed:
            try:
                observed[name] = call_in_spawned_process(
                    _observe_opt, name, timeout=START_TIMEOUT
                )
            except RuntimeError as error:
                observed[name] = error
        seen = observed[name]
        if isinstance(seen, RuntimeError):
            raise RuntimeError(f"scenario {name!r} did not start: {seen}")
        return seen

    return start


def _hooks(seen: dict) -> list[str]:
    return [hook for hook, _ in seen["calls"]]


def _infos(seen: dict, hook: str) -> list[dict]:
    return [info for called, info in seen["calls"] if called == hook]


# `call_in_spawned_process`, which the scenarios above start through.


def _double(value: int) -> int:
    return 2 * value


def _refuse(value: int) -> int:
    raise ValueError(f"refused {value}")


def _never_return(value: int) -> int:
    threading.Event().wait()
    return value


def test_call_in_spawned_process_returns_the_result():
    assert call_in_spawned_process(_double, 21, timeout=START_TIMEOUT) == 42


def test_call_in_spawned_process_raises_with_the_childs_traceback():
    with pytest.raises(RuntimeError, match="ValueError: refused 3"):
        call_in_spawned_process(_refuse, 3, timeout=START_TIMEOUT)


def test_call_in_spawned_process_ends_a_call_that_does_not_finish():
    """A call that outlasts its timeout raises, and its process is gone when
    the helper returns."""
    with pytest.raises(RuntimeError, match="did not finish within 1 s"):
        call_in_spawned_process(_never_return, 0, timeout=1)
    assert multiprocessing.active_children() == []


_GRAPH_HOOKS = {"capture_cudagraphs", "clear_cudagraphs"}

_NEEDS_DEVICE = pytest.mark.skipif(
    not cgu.current_platform.is_cuda_alike(), reason="requires CUDA or ROCm"
)


@_NEEDS_DEVICE
def test_estimate_calls_no_capture_hook_with_the_default_predicate(opt_started):
    """A model state that keeps the default predicate gets no capture call
    and no `clear_cudagraphs` call, although the estimate captured the
    runner's decoder graphs, and its inner graphs, over a temporary KV
    cache."""
    seen = opt_started("default-predicate")
    assert seen["cudagraph_mode"] != "NONE" and not seen["extensible_kv_cache"]
    assert seen["after_estimate"]["estimate"] > 0
    assert "initialize_kv_cache" in _hooks(seen)
    assert "capture_inner_cudagraphs" in _hooks(seen)
    assert not _GRAPH_HOOKS & set(_hooks(seen))


@_NEEDS_DEVICE
def test_estimate_calls_no_capture_hook_for_a_runner_with_its_own_capture(
    opt_started,
):
    """For a runner that replaces `capture_model` and
    `needs_cudagraph_capture`, and whose predicate returns `False`, the
    estimate allocates its temporary KV cache, captures nothing and tears
    down. No graph-capture hook is called, although the model state's
    predicate would return `True`."""
    seen = opt_started("own-capture-runner")
    assert seen["cudagraph_mode"] != "NONE" and not seen["extensible_kv_cache"]
    assert seen["after_estimate"]["own_capture_runner"] is True
    assert seen["after_estimate"]["estimate"] == 0
    assert "initialize_kv_cache" in _hooks(seen)
    assert not (_GRAPH_HOOKS | {"needs_cudagraph_capture"}) & set(_hooks(seen))


@_NEEDS_DEVICE
def test_estimate_captures_model_state_graphs_after_the_decoder_graphs(opt_started):
    """With decoder graphs to capture, `capture_model` calls
    `capture_cudagraphs` once, with `profile_only=True`, after the decoder
    graphs and the model state's inner graphs."""
    seen = opt_started("capturing")
    (capture,) = _infos(seen, "capture_cudagraphs")
    assert capture["profile_only"] is True
    assert capture["graphs_captured"] > 0
    hooks = _hooks(seen)
    assert hooks.index("capture_inner_cudagraphs") < hooks.index("capture_cudagraphs")


@_NEEDS_DEVICE
def test_capture_result_includes_what_the_model_state_call_added(opt_started):
    """The call precedes `capture_model`'s last free-memory reading: the
    graph memory it returns exceeds what the same capture returns without a
    model-state call by the 64 MiB the call kept, less an allowance of 8 MiB
    for allocator rounding between the two starts."""
    seen = opt_started("capturing")
    (capture,) = _infos(seen, "capture_cudagraphs")
    assert capture["reserved_bytes_taken"] >= 64 * MiB
    assert capture["device_bytes_taken"] >= 64 * MiB
    ((profile_only, captured),) = seen["capture_results"]
    ((_, without_call),) = opt_started("default-predicate")["capture_results"]
    assert profile_only is True
    # Both figures, for the record of a failure.
    assert captured - without_call >= (64 - 8) * MiB, (captured, without_call)


@_NEEDS_DEVICE
def test_capture_call_follows_the_speculator_graphs(opt_started):
    """With a draft-model speculator, `capture_model` calls
    `capture_cudagraphs` after the speculator's capture has completed."""
    hooks = _hooks(opt_started("capturing-speculator"))
    assert hooks.count("speculator_capture") == 1
    assert hooks.count("capture_cudagraphs") == 1
    assert hooks.index("speculator_capture") < hooks.index("capture_cudagraphs")


@_NEEDS_DEVICE
def test_encoder_graphs_alone_make_the_runner_capture(opt_started):
    """Under `cudagraph_mode` NONE, a model state with encoder graphs and the
    default predicate makes the runner report graphs to capture: the estimate
    allocates its temporary KV cache and captures, and calls no graph hook of
    the model state."""
    seen = opt_started("encoder-default-predicate")
    assert seen["cudagraph_mode"] == "NONE"
    assert seen["after_estimate"]["runner_needs_capture"] is True
    assert seen["after_estimate"]["estimate"] > 0
    assert "initialize_kv_cache" in _hooks(seen)
    assert not _GRAPH_HOOKS & set(_hooks(seen))


@_NEEDS_DEVICE
def test_capture_call_follows_the_encoder_graphs(opt_started):
    """`capture_model` calls `capture_cudagraphs` once the encoder graphs are
    captured, also when no decoder graph is captured."""
    seen = opt_started("encoder-capturing")
    assert seen["cudagraph_mode"] == "NONE"
    (capture,) = _infos(seen, "capture_cudagraphs")
    assert capture["encoder_graphs_captured"] is True
    assert "capture_inner_cudagraphs" not in _hooks(seen)


@_NEEDS_DEVICE
def test_capture_without_state_bindings_runs_without_a_window(opt_started):
    """A model state without persistent state has no bindings, so its capture
    call opens no startup-access window; the call runs and the estimate
    completes."""
    seen = opt_started("capturing")
    assert seen["error"] is None
    (capture,) = _infos(seen, "capture_cudagraphs")
    assert capture["has_bindings"] is False
    assert capture["stream_is_current"] is True
    assert seen["after_estimate"]["runner_has_bindings"] is False


@_NEEDS_DEVICE
def test_estimate_clears_a_model_state_without_persistent_state_once(opt_started):
    """The estimate's teardown calls `clear_cudagraphs` once, after the
    capture, for a model state without persistent state too."""
    hooks = _hooks(opt_started("capturing"))
    assert hooks.count("clear_cudagraphs") == 1
    assert hooks.index("capture_cudagraphs") < hooks.index("clear_cudagraphs")


@_NEEDS_DEVICE
def test_capture_call_is_inside_the_dummy_lora_scope(opt_started):
    """With LoRA enabled, the dummy LoRA of the capture is loaded while
    `capture_cudagraphs` runs and removed when `capture_model` returns."""
    seen = opt_started("capturing-lora")
    (capture,) = _infos(seen, "capture_cudagraphs")
    assert capture["dummy_loras"] == [1]
    assert seen["after_estimate"]["loras"] == []


@_NEEDS_DEVICE
def test_estimate_clears_when_the_capture_call_was_not_reached(opt_started):
    """When the runner's own capture raises before `capture_cudagraphs` is
    reached, here in the model state's inner graphs, that exception fails the
    estimate and the teardown still calls `clear_cudagraphs` once."""
    seen = opt_started("inner-capture-fails")
    assert seen["error"] == dict(type="_HookError", raised=0)
    hooks = _hooks(seen)
    assert "capture_cudagraphs" not in hooks
    assert hooks.count("clear_cudagraphs") == 1
    assert hooks.index("capture_inner_cudagraphs") < hooks.index("clear_cudagraphs")


@_NEEDS_DEVICE
def test_runner_capture_failure_outranks_a_failed_release(opt_started):
    """When the runner's own capture raised and `clear_cudagraphs` then
    raises too, the capture's exception is the one that propagates, and the
    exception from `clear_cudagraphs` is logged with its traceback."""
    seen = opt_started("inner-capture-and-release-fail")
    assert seen["error"] == dict(type="_HookError", raised=0)
    assert _hooks(seen).count("clear_cudagraphs") == 1
    assert seen["logged_raised"] == [1]


@_NEEDS_DEVICE
def test_non_state_start_captures_then_calls_post_warmup_once(opt_started):
    """A model without persistent state whose state captures gets, in the
    worker's warmup, one `profile_only=False` capture, then the sampler
    warmup, then `post_warmup` once, with the workspace locked."""
    seen = opt_started("capturing-start")
    startup = seen["calls"][: seen["startup_calls"]]
    hooks = [hook for hook, _ in startup]
    serving = [
        index
        for index, (hook, info) in enumerate(startup)
        if hook == "capture_cudagraphs" and not info["profile_only"]
    ]
    assert len(serving) == 1
    assert hooks.count("post_warmup") == 1
    last_sampler_warmup = len(hooks) - 1 - hooks[::-1].index("dummy_sampler_run")
    assert serving[0] < last_sampler_warmup < hooks.index("post_warmup")
    assert _infos(seen, "post_warmup")[0]["workspace_locked"] is True


@_NEEDS_DEVICE
def test_non_state_shutdown_calls_the_hook_once(opt_started):
    """Shutting the engine down calls `shutdown` once on a model state
    without persistent state, after startup."""
    seen = opt_started("capturing-start")
    assert _hooks(seen)[seen["startup_calls"] :].count("shutdown") == 1
    assert _hooks(seen)[: seen["startup_calls"]].count("shutdown") == 0


@_NEEDS_DEVICE
def test_pooling_start_calls_post_warmup_once(opt_started):
    """A pooling model's warmup, which warms no sampler up, still calls
    `post_warmup` once."""
    seen = opt_started("pooling-start")
    hooks = _hooks(seen)
    assert "dummy_sampler_run" not in hooks
    assert hooks[: seen["startup_calls"]].count("post_warmup") == 1


@_NEEDS_DEVICE
def test_pooling_shutdown_calls_the_hook_once(opt_started):
    """Shutting a pooling engine down calls `shutdown` once on its model
    state."""
    assert _hooks(opt_started("pooling-start")).count("shutdown") == 1
