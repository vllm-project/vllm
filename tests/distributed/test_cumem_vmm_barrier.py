# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the cross-rank quiesce barrier the cumem allocator issues at
sleep entry / wake_up exit (fix for vllm-project/vllm#45519).

These tests deliberately avoid the GPU path: they exercise the
``_quiesce_distributed_before_vmm_mutation`` helper directly under several
mock world-states, asserting that the right primitives are called (or
correctly skipped) on each path. The real PP-broadcast crash that motivates
the fix requires multi-rank NCCL and a sleep-mode-capable backend
(``cumem_tag`` from #45398) — that integration smoke is left to the
``tests/basic_correctness/test_mem.py`` suite running under the
hardware-gated multi-GPU CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# Module under test. Import lazily so the test file is collectable even if
# the cumem C extension isn't built (e.g. on macOS dev machines): the
# fallback path in vllm.device_allocator.cumem leaves the API surface
# defined even when ``cumem_available`` is False.
from vllm.device_allocator import cumem as cumem_module


class _Recorder:
    """Records sequence of calls so tests can assert ordering."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, name: str) -> Any:
        def _inner(*args, **kwargs):
            self.calls.append(name)

        return _inner


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


def _make_allocator() -> cumem_module.CuMemAllocator:
    """Construct an allocator without invoking the C-extension singleton.

    The class is a singleton via ``get_instance``; for unit tests we want
    a fresh instance each time without poisoning module-level state, so we
    instantiate directly. ``__init__`` does not touch the C extension —
    only the malloc / free callbacks invoke it lazily.
    """
    return cumem_module.CuMemAllocator()


def test_quiesce_noop_when_torch_distributed_unavailable(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """When ``torch.distributed.is_available()`` returns False (e.g. CPU-only
    PyTorch builds), the quiesce helper must not raise and must not attempt
    to look up a world group."""
    allocator = _make_allocator()

    # Force the cuda.synchronize call to record but no-op.
    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: False
    )
    # If we ever called barrier, it would error in this CPU-only env.
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: False
    )

    # Should return cleanly.
    allocator._quiesce_distributed_before_vmm_mutation()


def test_quiesce_noop_when_torch_distributed_uninitialized(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """When ``torch.distributed`` is available but not initialized (no
    ``init_process_group`` call), the helper must skip the barrier."""
    allocator = _make_allocator()

    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_initialized", lambda: False
    )

    # If the helper attempted to call barrier in this state torch would raise;
    # passing here proves the early-return fired.
    allocator._quiesce_distributed_before_vmm_mutation()


def test_quiesce_noop_when_kill_switch_disabled(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """When ``_ENABLE_BARRIER_FOR_VMM_MUTATION`` is set False, the helper
    must short-circuit before reaching the barrier — even with a fully-
    initialized distributed environment."""
    allocator = _make_allocator()

    # Stub out the parallel_state module import path.
    fake_parallel_state = SimpleNamespace(
        _ENABLE_BARRIER_FOR_VMM_MUTATION=False,
        get_world_group=lambda: pytest.fail(
            "get_world_group must not be called when the kill switch is off"
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.distributed.parallel_state",
        fake_parallel_state,
    )
    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_initialized", lambda: True
    )

    allocator._quiesce_distributed_before_vmm_mutation()


def test_quiesce_swallows_world_group_lookup_failure(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """If ``get_world_group`` raises (e.g. allocator used outside an inference
    engine), the helper must log and continue rather than crash the caller."""
    allocator = _make_allocator()

    def _raise() -> Any:
        raise AssertionError("world group not initialized")

    fake_parallel_state = SimpleNamespace(
        _ENABLE_BARRIER_FOR_VMM_MUTATION=True,
        get_world_group=_raise,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.distributed.parallel_state",
        fake_parallel_state,
    )
    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_initialized", lambda: True
    )

    # Must NOT raise.
    allocator._quiesce_distributed_before_vmm_mutation()


def test_quiesce_calls_barrier_on_world_cpu_group(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """Happy path: with everything wired, the helper must (a) sync cuda and
    (b) issue a CPU-side ``barrier`` on the world group's ``cpu_group``,
    in that order. The CPU group (not the device group) is deliberate —
    NCCL is the subsystem whose buffer registrations are in flux during a
    cumem_tag sleep/wake, so we keep our coordination primitive off it."""
    allocator = _make_allocator()

    sentinel_cpu_group = object()

    def _record_synchronize(*args, **kwargs) -> None:
        recorder.calls.append("cuda.synchronize")

    def _record_barrier(*args, **kwargs) -> None:
        # Assert we got the world's CPU group, not its device group.
        assert kwargs.get("group") is sentinel_cpu_group, (
            "quiesce must barrier on CPU group, not device group"
        )
        recorder.calls.append("dist.barrier")

    fake_world = SimpleNamespace(
        cpu_group=sentinel_cpu_group,
        device_group=object(),  # distinct sentinel — must NOT be used
    )
    fake_parallel_state = SimpleNamespace(
        _ENABLE_BARRIER_FOR_VMM_MUTATION=True,
        get_world_group=lambda: fake_world,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.distributed.parallel_state",
        fake_parallel_state,
    )
    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.cuda, "synchronize", _record_synchronize
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "barrier", _record_barrier
    )

    allocator._quiesce_distributed_before_vmm_mutation()

    # Order matters: drain device-side work before issuing the cross-rank
    # ordering primitive, so any in-flight NCCL kernels have completed
    # before other ranks proceed past the barrier.
    assert recorder.calls == ["cuda.synchronize", "dist.barrier"], (
        f"Expected synchronize then barrier; got {recorder.calls}"
    )


def test_quiesce_swallows_barrier_failure(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """Barrier failures must not mask the originating sleep/wake call —
    they degrade coordination but should never become the user-visible
    exception over the actual VMM mutation."""
    allocator = _make_allocator()

    sentinel_cpu_group = object()
    fake_world = SimpleNamespace(cpu_group=sentinel_cpu_group)
    fake_parallel_state = SimpleNamespace(
        _ENABLE_BARRIER_FOR_VMM_MUTATION=True,
        get_world_group=lambda: fake_world,
    )

    def _raising_barrier(*args, **kwargs) -> None:
        raise RuntimeError("simulated NCCL/gloo failure")

    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.distributed.parallel_state",
        fake_parallel_state,
    )
    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "barrier", _raising_barrier
    )

    # Must NOT raise.
    allocator._quiesce_distributed_before_vmm_mutation()


def test_quiesce_swallows_synchronize_failure(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """``torch.cuda.synchronize`` errors must also degrade-not-fail. Symmetric
    to the barrier-failure case above."""
    allocator = _make_allocator()

    def _raising_synchronize(*args, **kwargs) -> None:
        raise RuntimeError("simulated CUDA context fault")

    monkeypatch.setattr(
        cumem_module.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cumem_module.torch.cuda, "synchronize", _raising_synchronize
    )
    monkeypatch.setattr(
        cumem_module.torch.distributed, "is_available", lambda: False
    )

    # Must NOT raise.
    allocator._quiesce_distributed_before_vmm_mutation()


def test_kill_switch_setter_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """``set_enable_cumem_vmm_barrier`` toggles the module-level flag in
    both directions and returns the value the getter sees."""
    from vllm.distributed import parallel_state

    original = parallel_state._ENABLE_BARRIER_FOR_VMM_MUTATION
    try:
        parallel_state.set_enable_cumem_vmm_barrier(False)
        assert parallel_state._ENABLE_BARRIER_FOR_VMM_MUTATION is False

        parallel_state.set_enable_cumem_vmm_barrier(True)
        assert parallel_state._ENABLE_BARRIER_FOR_VMM_MUTATION is True
    finally:
        parallel_state._ENABLE_BARRIER_FOR_VMM_MUTATION = original
