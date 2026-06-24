# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the device-side wake synchronization in
``CuMemAllocator.wake_up`` (part of the fix for issue #45519).

The fix adds a ``torch.cuda.synchronize()`` at the *end* of
``CuMemAllocator.wake_up`` — after the per-region ``create_and_map`` re-map
loop and the ``cudaMemcpy`` restores, and before the method returns to the
caller (``Worker.wake_up``).

This synchronize is a purely LOCAL guarantee: it drains *this* rank's device
work (the freshly re-mapped VMM regions) so the rank does not report wake
success into the cross-rank handshake while its own remaps are still pending on
the device. It provides NO cross-rank ordering on its own — the cross-rank
guarantee for the #45519 PP-broadcast race lives in ``Worker.wake_up`` (the
gloo all-reduce of a wake-success flag).

Two accuracy notes (the original rationale for this synchronize was wrong on
both, and a maintainer would bounce it):
  * The ``cudaMemcpy`` restores use the *synchronous*, host-blocking
    ``cudaMemcpy`` (see ``CudaRTLibrary``), NOT ``cudaMemcpyAsync`` — so the
    copies have already completed w.r.t. the host by the time the loop exits;
    they are not what the synchronize guards.
  * The ``sleep()`` offload loop does NOT ``torch.cuda.synchronize`` per handle
    between its D2H ``cudaMemcpy`` and the following ``unmap_and_release`` — the
    only per-handle sync lives in ``_python_free_callback`` (a different path).
    So this is not a "wake-side analogue" of a sleep-side per-handle sync.

LIMITATION (documented, per the task): a *true* unit test of ``wake_up`` is not
feasible GPU-free — ``create_and_map`` and ``libcudart.cudaMemcpy`` drive real
CUDA driver / VMM calls, and ``torch.cuda.synchronize`` requires a CUDA context.
This is the closest mockable test: we construct a real ``CuMemAllocator`` (its
``__init__`` is pure-Python — only dict assignment), populate
``pointer_to_data`` with one real ``AllocationData`` carrying a CPU backup
tensor (so the ``cudaMemcpy`` restore branch is exercised), and mock the three
CUDA-touching surfaces (``create_and_map``, the module ``libcudart``, and
``torch.cuda.synchronize``). We then assert ``synchronize`` was called and that
it was called AFTER the remap/restore work (call-order), which is the property
the fix guarantees.
"""

from unittest import mock

import pytest

from vllm.device_allocator import AllocationData
from vllm.device_allocator import cumem as cumem_mod
from vllm.device_allocator.cumem import CuMemAllocator


def _make_allocator_with_one_region(calls: list[str]):
    """A real CuMemAllocator with one offloaded region queued for wake.

    `calls` records the load-bearing event order: each per-region remap
    (`create_and_map`) / restore (`cudaMemcpy`) appends, and the final
    `torch.cuda.synchronize` appends `"synchronize"`.
    """
    allocator = CuMemAllocator.__new__(CuMemAllocator)
    allocator.pointer_to_data = {}
    allocator.current_tag = CuMemAllocator.default_tag
    allocator.allocator_and_pools = {}

    # One region: handle = (device, size, ptr, vmm_handle). Give it a CPU
    # backup tensor so the cudaMemcpy restore branch runs (the host-blocking
    # restore the synchronize sequences the VMM remap against).
    cpu_backup = mock.MagicMock()
    cpu_backup.numel.return_value = 4
    cpu_backup.element_size.return_value = 2
    cpu_backup.data_ptr.return_value = 0xCAFE
    ptr = 0x1000
    handle = (0, 8, ptr, 0xABCD)
    allocator.pointer_to_data[ptr] = AllocationData(
        handle=handle, tag=CuMemAllocator.default_tag, cpu_backup_tensor=cpu_backup
    )
    return allocator


def _run_wake(allocator, calls):
    fake_libcudart = mock.MagicMock()
    fake_libcudart.cudaMemcpy.side_effect = lambda *a, **k: calls.append("cudaMemcpy")

    with mock.patch.object(
        cumem_mod, "create_and_map",
        side_effect=lambda *a, **k: calls.append("create_and_map"),
    ), mock.patch.object(cumem_mod, "libcudart", fake_libcudart), mock.patch(
        "torch.cuda.synchronize",
        side_effect=lambda *a, **k: calls.append("synchronize"),
    ) as sync:
        allocator.wake_up()
    return sync


def test_wake_up_calls_cuda_synchronize():
    """`CuMemAllocator.wake_up` must call `torch.cuda.synchronize()`."""
    calls: list[str] = []
    allocator = _make_allocator_with_one_region(calls)
    sync = _run_wake(allocator, calls)

    sync.assert_called_once()
    assert "synchronize" in calls


def test_wake_up_synchronize_runs_after_remap_and_restore():
    """The synchronize must come AFTER the remap (`create_and_map`) and the
    `cudaMemcpy` restore, and is the last device op before return.

    A synchronize placed *inside* / *before* the loop — or omitted — would
    leave this rank reporting wake success while its VMM remaps are still
    pending on the device. We assert ordering explicitly.
    """
    calls: list[str] = []
    allocator = _make_allocator_with_one_region(calls)
    _run_wake(allocator, calls)

    assert calls == ["create_and_map", "cudaMemcpy", "synchronize"], (
        "expected remap -> cudaMemcpy restore -> device synchronize, got "
        f"{calls}"
    )
    # Synchronize is the final device action before wake_up returns.
    assert calls[-1] == "synchronize"


def test_wake_up_synchronizes_even_with_no_backup_tensor():
    """Even a region with no CPU backup (re-map only, no cudaMemcpy) must
    still be followed by the device synchronize before returning."""
    calls: list[str] = []
    allocator = CuMemAllocator.__new__(CuMemAllocator)
    allocator.pointer_to_data = {}
    allocator.current_tag = CuMemAllocator.default_tag
    allocator.allocator_and_pools = {}
    ptr = 0x2000
    allocator.pointer_to_data[ptr] = AllocationData(
        handle=(0, 8, ptr, 0xBEEF),
        tag=CuMemAllocator.default_tag,
        cpu_backup_tensor=None,
    )
    sync = _run_wake(allocator, calls)

    sync.assert_called_once()
    assert calls == ["create_and_map", "synchronize"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
