# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the CuMemAllocator.sleep() in-flight-kernel drain.

Under async scheduling (``max_concurrent_batches > 1``),
``EngineCore.step_with_batch_queue`` submits ``execute_model(non_block=True)``
and returns *without* awaiting the returned future. A subsequent ``/sleep``
reaches ``CuMemAllocator.sleep``, which loops over allocations calling
``unmap_and_release`` -> ``cuMemUnmap``, tearing down the KV/weight virtual
address space while the still-running forward kernel is writing to it. That
surfaces as an Xid-31 write-to-unmapped-VA fault and kills EngineCore.

The fix is a device-wide ``torch.cuda.synchronize()`` at the top of
``sleep()``, before the unmap loop, to drain all in-flight kernels. This test
asserts that ordering invariant without needing a GPU: it monkeypatches
``torch.cuda.synchronize`` and the module-level ``unmap_and_release`` to record
the order in which they are called, then drives ``sleep()`` against a single
fake allocation.

Pre-fix (no synchronize before the unmap loop) this test FAILS; post-fix it
PASSES. The full async-drain GPU repro (start a long forward under
``max_concurrent_batches=2``, fire ``/sleep`` mid-step, assert no Xid-31 /
CUDA_ERROR_ILLEGAL_ADDRESS) is the integration variant; this unit assertion is
the carryable proof.
"""

import pytest

cumem_module = pytest.importorskip("vllm.device_allocator.cumem")

from vllm.device_allocator import AllocationData  # noqa: E402
from vllm.device_allocator.cumem import CuMemAllocator  # noqa: E402


def _make_allocator_with_one_allocation():
    """Build a CuMemAllocator carrying a single fake allocation.

    ``__init__`` has no GPU dependency (it only sets up dicts + bound-method
    references), so we can construct it directly on a CPU host. The handle is
    ``(py_device, py_size, py_ptr, py_handle)``.
    """
    allocator = CuMemAllocator()
    # device 0, 4096 bytes, ptr 0x1000, opaque handle 0x2000
    handle = (0, 4096, 0x1000, 0x2000)
    allocator.pointer_to_data[handle[2]] = AllocationData(
        handle=handle, tag=CuMemAllocator.default_tag
    )
    return allocator


def test_sleep_drains_kernels_before_unmap(monkeypatch):
    """synchronize() must be called before any unmap_and_release()."""
    calls: list[str] = []

    import torch

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a, **k: calls.append("synchronize")
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda *a, **k: None
    )
    # Stub the C-extension teardown so no real CUDA work is attempted.
    monkeypatch.setattr(
        cumem_module, "unmap_and_release", lambda handle: calls.append("unmap")
    )

    allocator = _make_allocator_with_one_allocation()
    # offload_tags=() => discard everything (no cudaMemcpy backup path), which
    # exercises the unmap loop without touching libcudart.
    allocator.sleep(offload_tags=())

    assert "synchronize" in calls, (
        "CuMemAllocator.sleep must drain in-flight kernels via "
        "torch.cuda.synchronize() before unmapping VA"
    )
    assert "unmap" in calls, "test did not exercise the unmap loop"
    assert calls.index("synchronize") < calls.index("unmap"), (
        "torch.cuda.synchronize() must be called BEFORE any unmap_and_release() "
        "/ cuMemUnmap; otherwise an in-flight forward kernel (async scheduling, "
        "max_concurrent_batches > 1) races VA teardown -> Xid-31 fault"
    )
