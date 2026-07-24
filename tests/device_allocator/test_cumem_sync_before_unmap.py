# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for the cumem-allocator stream-synchronization invariant
that prevents `cudaErrorIllegalAddress` on rapid `/sleep` after `/wake_up`
or while in-flight decode kernels are still running.

Bug history:
  - https://github.com/vllm-project/vllm/issues/45520
    `/sleep` crashes engine with CUDA illegal-access if in-flight decode
    requests exist (no drain semantics).
  - https://github.com/vllm-project/vllm/issues/36753
    POST /wake_up causes vLLM process to crash. 500 Internal Server Error.

Invariants this file asserts:
  1. `CuMemAllocator.sleep()` calls `torch.cuda.synchronize()` BEFORE any
     `cuMemUnmap` or D2H `cudaMemcpy`. Without this, kernels still in
     flight (decode steps, P2P sends, the H2D `cudaMemcpy` from a prior
     `wake_up`) race the unmap and surface as
     `CUDART error: an illegal memory access was encountered`.
  2. `CuMemAllocator.wake_up()` calls `torch.cuda.synchronize()` BEFORE
     returning, so a subsequent rapid `/sleep` cannot race the tail of the
     restore copies.

These tests do NOT require a GPU — they patch the cumem C-extension entry
points and `torch.cuda.synchronize` to record call order, then exercise
the Python sleep/wake state machine and assert ordering.
"""

from __future__ import annotations

import pytest
import torch

from vllm.device_allocator import AllocationData
from vllm.device_allocator import cumem as cumem_mod


@pytest.fixture
def fake_cumem(monkeypatch: pytest.MonkeyPatch):
    """
    Stub the cumem C-extension entry points + libcudart + torch.cuda.synchronize
    so the sleep/wake state machine can be exercised on hosts without CUDA.
    Records every call into ``calls`` (a list of (op_name, args) tuples)
    so tests can assert ordering.
    """
    calls: list[tuple[str, tuple]] = []

    def fake_unmap_and_release(handle):
        calls.append(("unmap_and_release", (handle,)))

    def fake_create_and_map(handle):
        calls.append(("create_and_map", (handle,)))

    class FakeLibCudart:
        def cudaMemcpy(self, dst, src, n):  # noqa: N802 — match real API
            calls.append(("cudaMemcpy", (dst, src, n)))

    def fake_synchronize(*args, **kwargs):
        calls.append(("torch.cuda.synchronize", args))

    # Force "available" path even on hosts without the C-extension
    monkeypatch.setattr(cumem_mod, "cumem_available", True, raising=False)
    monkeypatch.setattr(cumem_mod, "unmap_and_release", fake_unmap_and_release)
    monkeypatch.setattr(cumem_mod, "create_and_map", fake_create_and_map)
    monkeypatch.setattr(cumem_mod, "libcudart", FakeLibCudart())
    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)

    # Reset the singleton so each test gets a clean instance
    monkeypatch.setattr(cumem_mod.CuMemAllocator, "instance", None, raising=False)
    return calls


def _seed_allocations(allocator: cumem_mod.CuMemAllocator, n: int = 3) -> None:
    """Seed the allocator with N fake allocations under the default tag."""
    for i in range(n):
        ptr = 0x1000 + i * 0x100
        # AllocationData(handle, tag) — handle shape matches the real
        # (device_id, size, py_d_mem, ...) tuple used by python_create_and_map.
        handle = (0, 4096, ptr)
        allocator.pointer_to_data[ptr] = AllocationData(
            handle, cumem_mod.CuMemAllocator.default_tag
        )


def _first_index(calls: list[tuple[str, tuple]], op: str) -> int:
    for i, (name, _) in enumerate(calls):
        if name == op:
            return i
    raise AssertionError(f"{op!r} never called; calls={calls!r}")


# ---------------------------------------------------------------------------
# Invariant 1: sleep() syncs BEFORE any unmap or D2H cudaMemcpy.
# ---------------------------------------------------------------------------


def test_sleep_synchronizes_before_unmap(fake_cumem):
    """The CUDA stream sync must happen before the first unmap_and_release."""
    allocator = cumem_mod.CuMemAllocator.get_instance()
    _seed_allocations(allocator, n=3)

    allocator.sleep(offload_tags=tuple())  # discard-all path

    sync_idx = _first_index(fake_cumem, "torch.cuda.synchronize")
    unmap_idx = _first_index(fake_cumem, "unmap_and_release")
    assert sync_idx < unmap_idx, (
        f"sleep() must sync before any unmap (found sync@{sync_idx} "
        f"unmap@{unmap_idx}); calls={fake_cumem!r}"
    )


def test_sleep_synchronizes_before_d2h_copy(fake_cumem):
    """For the offload path, sync must precede the D2H cudaMemcpy too."""
    allocator = cumem_mod.CuMemAllocator.get_instance()
    _seed_allocations(allocator, n=2)

    allocator.sleep()  # default tag → all allocations get offloaded (D2H copy)

    sync_idx = _first_index(fake_cumem, "torch.cuda.synchronize")
    memcpy_idx = _first_index(fake_cumem, "cudaMemcpy")
    assert sync_idx < memcpy_idx, (
        f"sleep() must sync before D2H cudaMemcpy (found sync@{sync_idx} "
        f"memcpy@{memcpy_idx}); calls={fake_cumem!r}"
    )


def test_sleep_with_no_allocations_still_safe(fake_cumem):
    """Sleep with empty pool is a no-op; sync may be issued, must not crash."""
    allocator = cumem_mod.CuMemAllocator.get_instance()
    # No allocations seeded.
    allocator.sleep(offload_tags=tuple())
    # No unmap/memcpy calls expected
    assert not any(
        op in ("unmap_and_release", "cudaMemcpy") for op, _ in fake_cumem
    ), fake_cumem


# ---------------------------------------------------------------------------
# Invariant 2: wake_up() syncs BEFORE returning.
# ---------------------------------------------------------------------------


def test_wake_up_synchronizes_after_h2d_copies(fake_cumem):
    """The final sync after wake_up's H2D copies must happen before return."""
    allocator = cumem_mod.CuMemAllocator.get_instance()
    _seed_allocations(allocator, n=3)
    # Pretend we've already slept — assign cpu_backup_tensors so wake_up
    # restores via cudaMemcpy (the realistic post-sleep state).
    for data in allocator.pointer_to_data.values():
        data.cpu_backup_tensor = torch.empty(4096, dtype=torch.uint8, device="cpu")

    allocator.wake_up()

    # The sync() call must be the LAST recorded synchronize, and it must
    # come after the H2D cudaMemcpy operations.
    sync_indices = [i for i, (op, _) in enumerate(fake_cumem) if op == "torch.cuda.synchronize"]
    memcpy_indices = [i for i, (op, _) in enumerate(fake_cumem) if op == "cudaMemcpy"]
    assert sync_indices, f"wake_up() must call synchronize; calls={fake_cumem!r}"
    assert memcpy_indices, f"wake_up() must memcpy with backups; calls={fake_cumem!r}"
    assert sync_indices[-1] > memcpy_indices[-1], (
        f"wake_up() final sync must follow last cudaMemcpy "
        f"(sync_idxs={sync_indices} memcpy_idxs={memcpy_indices}); calls={fake_cumem!r}"
    )


# ---------------------------------------------------------------------------
# End-to-end: rapid sleep-after-wake (the production failure shape).
# ---------------------------------------------------------------------------


def test_rapid_sleep_after_wake_up_serializes_through_sync(fake_cumem):
    """
    Production failure shape from #45520:

        wake_up() → 200 → caller fires sleep() within ~2-9s → CUDART
        illegal memory access on the offload-loop's cudaMemcpy/unmap.

    With both fixes in place, every wake_up→sleep transition is guarded by
    two synchronize() calls (one closing wake_up, one opening sleep), so
    no in-flight CUDA work can race the next state's unmap/copy.
    """
    allocator = cumem_mod.CuMemAllocator.get_instance()
    _seed_allocations(allocator, n=2)
    for data in allocator.pointer_to_data.values():
        data.cpu_backup_tensor = torch.empty(4096, dtype=torch.uint8, device="cpu")

    # 1) wake_up → must end with a sync.
    allocator.wake_up()
    syncs_after_wake = [i for i, (op, _) in enumerate(fake_cumem) if op == "torch.cuda.synchronize"]
    assert syncs_after_wake, f"wake_up missing sync; calls={fake_cumem!r}"

    boundary = len(fake_cumem)

    # 2) Rapid sleep — must START with a sync (before any unmap).
    allocator.sleep(offload_tags=tuple())
    sleep_calls = fake_cumem[boundary:]
    sleep_sync_idx = _first_index(sleep_calls, "torch.cuda.synchronize")
    sleep_unmap_idx = _first_index(sleep_calls, "unmap_and_release")
    assert sleep_sync_idx < sleep_unmap_idx, (
        f"rapid sleep after wake_up must sync before unmap; "
        f"sleep_calls={sleep_calls!r}"
    )

    # 3) The two syncs (close-wake, open-sleep) sit on either side of the
    #    boundary — no unmap/memcpy without a preceding sync exists in
    #    either half.
    assert any(
        op == "torch.cuda.synchronize" for op, _ in fake_cumem[:boundary]
    ), "wake_up half missing sync"
    assert any(
        op == "torch.cuda.synchronize" for op, _ in fake_cumem[boundary:]
    ), "sleep half missing sync"
