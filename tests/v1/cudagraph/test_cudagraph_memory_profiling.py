# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `GPUModelRunner.profile_cudagraph_memory`.

The estimate this returns is subtracted from the KV cache budget, so an
undercount is not conservative: whatever it misses gets handed to the KV cache
and pushes total usage past `gpu_memory_utilization`.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker import gpu_model_runner as gmr

pytestmark = pytest.mark.cpu_test

MIB = 1 << 20
GIB = 1 << 30
TOTAL_MEMORY = 80 * GIB


class _FakeMemory:
    """Stand-in for the device allocator's free-memory counter."""

    def __init__(self, free: int = 64 * GIB):
        self.free = free

    def get_memory_info(self) -> tuple[int, int]:
        return self.free, TOTAL_MEMORY

    def consume(self, nbytes: int) -> None:
        self.free -= nbytes


class _FakeWrapper:
    """Minimal stand-in for CUDAGraphWrapper, to check pool save/restore."""

    def __init__(self, graph_pool: object):
        self.graph_pool = graph_pool
        self.cleared = False

    def clear_graphs(self) -> None:
        self.cleared = True


def _make_wrapper_class(instances: list[_FakeWrapper]):
    class _WrapperClass:
        _all_instances = instances

        @classmethod
        def clear_all_graphs(cls) -> None:
            for instance in cls._all_instances:
                instance.clear_graphs()

    return _WrapperClass


def _desc(num_tokens: int) -> BatchDescriptor:
    return BatchDescriptor(num_tokens=num_tokens, uniform=True)


class _FakeRunner:
    """Only the surface `profile_cudagraph_memory` actually touches."""

    def __init__(
        self,
        memory: _FakeMemory,
        capture_descs: list[tuple[CUDAGraphMode, list[BatchDescriptor]]],
        capture_cost: dict[CUDAGraphMode, int] | None = None,
        encoder_manager: object | None = None,
        fail_on_call: int | None = None,
        cleanup_release: int = 0,
        setup_cost: int = 0,
    ):
        self.memory = memory
        self._capture_descs = capture_descs
        self._capture_cost = capture_cost or {}
        self._encoder_manager = encoder_manager
        self._fail_on_call = fail_on_call
        self._cleanup_release = cleanup_release
        self._setup_cost = setup_cost

        self.captured: list[tuple[CUDAGraphMode, int, int | None]] = []
        self.kv_cache_initialized = False
        self.kv_cache_cleaned = False
        self.loras_removed = False
        self.cudagraph_profiling_retained_memory = 0

        self.device = torch.device("cpu")
        self.max_model_len = 4096
        self.max_num_tokens = 8192
        self.lora_config = None
        self.vllm_config = SimpleNamespace()
        self.cudagraph_dispatcher = SimpleNamespace(
            get_capture_descs=lambda: self._capture_descs,
            cudagraph_keys={mode: {object()} for mode, _ in capture_descs},
            keys_initialized=True,
        )

    def _init_minimal_kv_cache_for_profiling(self) -> None:
        self.kv_cache_initialized = True
        self.memory.consume(self._setup_cost)

    def _cleanup_profiling_kv_cache(self) -> None:
        self.kv_cache_cleaned = True
        self.memory.consume(-self._cleanup_release)

    def _create_encoder_cudagraph_manager(self):
        return self._encoder_manager

    def maybe_remove_all_loras(self, lora_config) -> None:
        self.loras_removed = True

    @contextmanager
    def _freeze_gc(self):
        yield

    def _warmup_and_capture(
        self, desc, cudagraph_runtime_mode, profile_seq_lens=None, **kwargs
    ) -> None:
        self.captured.append(
            (cudagraph_runtime_mode, desc.num_tokens, profile_seq_lens)
        )
        if self._fail_on_call is not None and len(self.captured) == self._fail_on_call:
            raise RuntimeError("capture blew up")
        self.memory.consume(self._capture_cost.get(cudagraph_runtime_mode, 0))


@pytest.fixture
def patched_env(monkeypatch):
    """Neutralize everything `profile_cudagraph_memory` reaches for globally."""
    wrappers: list[_FakeWrapper] = []

    @contextmanager
    def fake_set_current_vllm_config(config):
        yield

    @contextmanager
    def fake_graph_capture(device, graph_capture_context=None):
        yield SimpleNamespace()

    monkeypatch.setattr(gmr, "set_current_vllm_config", fake_set_current_vllm_config)
    monkeypatch.setattr(gmr, "graph_capture", fake_graph_capture)
    monkeypatch.setattr(gmr, "set_cudagraph_capturing_enabled", lambda enabled: None)
    monkeypatch.setattr(
        gmr,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: False,
            graph_pool_handle=lambda: object(),
        ),
    )
    monkeypatch.setattr(gmr, "CUDAGraphWrapper", _make_wrapper_class(wrappers))
    monkeypatch.setattr(gmr, "BreakableCUDAGraphWrapper", _make_wrapper_class([]))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda *a, **k: None)
    return SimpleNamespace(wrappers=wrappers, monkeypatch=monkeypatch)


def _run(runner: _FakeRunner, patched_env) -> int:
    patched_env.monkeypatch.setattr(
        torch.accelerator, "get_memory_info", runner.memory.get_memory_info
    )
    return gmr.GPUModelRunner.profile_cudagraph_memory(runner)


def test_profiles_every_descriptor(patched_env):
    """Every capture size must be profiled, not just the first two.

    Extrapolating from two samples assumes pool growth is linear in the number
    of graphs, which it is not.
    """
    descs = [_desc(n) for n in (128, 96, 64, 32, 16)]
    runner = _FakeRunner(
        _FakeMemory(),
        [(CUDAGraphMode.PIECEWISE, descs)],
        capture_cost={CUDAGraphMode.PIECEWISE: 100 * MIB},
    )

    estimate = _run(runner, patched_env)

    assert [tokens for _, tokens, _ in runner.captured] == [128, 96, 64, 32, 16]
    assert estimate == 5 * 100 * MIB


def test_estimate_spans_all_modes(patched_env):
    """PIECEWISE and FULL both land in the estimate."""
    runner = _FakeRunner(
        _FakeMemory(),
        [
            (CUDAGraphMode.PIECEWISE, [_desc(128), _desc(64)]),
            (CUDAGraphMode.FULL, [_desc(64), _desc(32)]),
        ],
        capture_cost={
            CUDAGraphMode.PIECEWISE: 400 * MIB,
            CUDAGraphMode.FULL: 100 * MIB,
        },
    )

    estimate = _run(runner, patched_env)

    assert {mode for mode, _, _ in runner.captured} == {
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
    }
    assert estimate == (2 * 400 + 2 * 100) * MIB


def test_negative_mode_delta_does_not_inflate_total(patched_env):
    """A mode measuring negative must not be clamped away in isolation.

    Clamping per mode and summing would report FULL's 2 GiB while the pool only
    grew 1.5 GiB, over-reserving the difference away from the KV cache.
    """
    runner = _FakeRunner(
        _FakeMemory(),
        [
            (CUDAGraphMode.PIECEWISE, [_desc(128)]),
            (CUDAGraphMode.FULL, [_desc(64)]),
        ],
        capture_cost={
            CUDAGraphMode.PIECEWISE: -512 * MIB,
            CUDAGraphMode.FULL: 2 * GIB,
        },
    )

    estimate = _run(runner, patched_env)

    assert estimate == 2 * GIB - 512 * MIB


def test_net_release_clamps_to_zero(patched_env):
    """Overall growth below zero reports zero, never a negative budget."""
    runner = _FakeRunner(
        _FakeMemory(),
        [(CUDAGraphMode.PIECEWISE, [_desc(128)])],
        capture_cost={CUDAGraphMode.PIECEWISE: -256 * MIB},
    )

    assert _run(runner, patched_env) == 0


def test_encoder_estimate_added_on_top(patched_env):
    """Encoder graphs use a separate pool, so they add rather than overlay."""
    memory = _FakeMemory()

    class _EncoderManager:
        token_budgets = [512]
        cleared = False

        def get_num_graphs_to_capture(self):
            return 2

        def capture(self, graph_pool):
            memory.consume(256 * MIB)

        def clear(self):
            self.cleared = True

    encoder_manager = _EncoderManager()
    runner = _FakeRunner(
        memory,
        [(CUDAGraphMode.PIECEWISE, [_desc(128)])],
        capture_cost={CUDAGraphMode.PIECEWISE: 1 * GIB},
        encoder_manager=encoder_manager,
    )

    assert _run(runner, patched_env) == 1 * GIB + 256 * MIB
    assert encoder_manager.cleared


def test_full_mode_passes_profile_seq_lens_only_for_first_descriptor(patched_env):
    runner = _FakeRunner(
        _FakeMemory(),
        [(CUDAGraphMode.FULL, [_desc(64), _desc(32), _desc(16)])],
    )

    _run(runner, patched_env)

    seq_lens = [seq_lens for _, _, seq_lens in runner.captured]
    assert seq_lens[0] is not None
    assert seq_lens[1:] == [None, None]


def test_no_graphs_returns_zero_and_cleans_up(patched_env):
    runner = _FakeRunner(_FakeMemory(), [])

    assert _run(runner, patched_env) == 0
    assert runner.kv_cache_cleaned


def test_estimate_covers_setup_done_before_capturing(patched_env):
    """Standing up the profiling KV cache initializes the attention backends
    and metadata builders. That memory is rebuilt for the real KV cache and is
    budgeted nowhere else, so leaving it out hands it to the KV cache."""
    runner = _FakeRunner(
        _FakeMemory(),
        [(CUDAGraphMode.PIECEWISE, [_desc(128)])],
        capture_cost={CUDAGraphMode.PIECEWISE: 1 * GIB},
        setup_cost=768 * MIB,
    )

    assert _run(runner, patched_env) == 1 * GIB + 768 * MIB


def test_retained_memory_excludes_what_cleanup_gave_back(patched_env):
    """Profiling keeps the scratch its captures allocated; the real capture
    reuses it, so it has to be reported separately from the estimate."""
    runner = _FakeRunner(
        _FakeMemory(),
        [(CUDAGraphMode.PIECEWISE, [_desc(128)])],
        capture_cost={CUDAGraphMode.PIECEWISE: 1 * GIB},
        cleanup_release=400 * MIB,
    )

    estimate = _run(runner, patched_env)

    assert estimate == 1 * GIB
    assert runner.cudagraph_profiling_retained_memory == 1 * GIB - 400 * MIB


def test_capture_failure_still_restores_state(patched_env):
    """Capture errors propagate, but profiling state must not leak."""
    original_pool = object()
    wrapper = _FakeWrapper(original_pool)
    patched_env.wrappers.append(wrapper)

    runner = _FakeRunner(
        _FakeMemory(),
        [(CUDAGraphMode.PIECEWISE, [_desc(128), _desc(64)])],
        fail_on_call=2,
    )

    with pytest.raises(RuntimeError, match="capture blew up"):
        _run(runner, patched_env)

    assert wrapper.graph_pool is original_pool
    assert wrapper.cleared
    assert runner.kv_cache_cleaned
    assert runner.loras_removed
    assert runner.cudagraph_dispatcher.keys_initialized is False
