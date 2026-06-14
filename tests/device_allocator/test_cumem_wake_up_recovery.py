# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``CuMemAllocator.wake_up`` recovery semantics.

These tests exercise the Python-layer behaviour of the cumem allocator's
wake-up loop without requiring a GPU. They drive the loop with a mocked C
extension (``python_create_and_map``) so we can deterministically simulate
the failure modes observed on multi-process TP/PP workloads:

* Transient wake-time ``cuMemMap`` failures that should be surfaced as a
  structured ``WakeUpPartialFailure`` rather than a bare ``RuntimeError``.
* Persistent failures that should leave ``pointer_to_data`` consistent so a
  subsequent retry can re-attempt only the failed entries.
* A failure on one entry must not silently abort the wake of the remaining
  entries — the loop must complete and report every failure together.

These tests would FAIL against the pre-fix wake_up loop because:
  1. A bare exception from the C call propagates out of the first failing
     iteration, leaving later allocations un-mapped with no signal that they
     were skipped.
  2. The exception type is the generic ``RuntimeError`` raised by the C
     extension, with no structured list of failed pointers.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest


# -----------------------------------------------------------------------------
# Module loader: stub the C extension + cuda wrapper so cumem.py imports
# cleanly on non-CUDA test runners (CI, contributor laptops). Without these
# stubs, ``from vllm.cumem_allocator import ...`` raises ModuleNotFoundError
# and the module sets ``cumem_available = False`` — which makes
# ``CuMemAllocator.get_instance()`` assert out before we can exercise it.
# -----------------------------------------------------------------------------


@pytest.fixture
def cumem_module(monkeypatch):
    """Import (or re-import) ``vllm.device_allocator.cumem`` with the C
    extension and CUDA wrapper stubbed out so the import succeeds without a
    GPU. Returns the freshly-imported module so tests can patch attributes
    on it directly."""

    # Stub C extension before import.
    fake_ext = mock.MagicMock(name="vllm.cumem_allocator")
    monkeypatch.setitem(sys.modules, "vllm.cumem_allocator", fake_ext)

    # Stub the CUDA wrapper module that cumem imports at top-level.
    fake_wrapper_mod = mock.MagicMock(
        name="vllm.distributed.device_communicators.cuda_wrapper"
    )
    fake_wrapper_mod.CudaRTLibrary = mock.MagicMock(name="CudaRTLibrary")
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.cuda_wrapper",
        fake_wrapper_mod,
    )

    # Stub find_loaded_library so it returns a fake path without scanning
    # /proc/self/maps for an actual loaded library.
    fake_sys_utils = mock.MagicMock(name="vllm.utils.system_utils")
    fake_sys_utils.find_loaded_library = mock.MagicMock(
        return_value="/fake/path/cumem_allocator.so"
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.system_utils", fake_sys_utils)

    # Force a fresh import so the top-level try/except sees our stubs.
    sys.modules.pop("vllm.device_allocator.cumem", None)
    import vllm.device_allocator.cumem as cumem  # noqa: E402

    # Reset the singleton between tests so state doesn't leak.
    cumem.CuMemAllocator.instance = None

    yield cumem

    # Cleanup after the test.
    cumem.CuMemAllocator.instance = None
    sys.modules.pop("vllm.device_allocator.cumem", None)


def _make_handle(device: int, size: int, d_mem: int, p_handle: int) -> tuple:
    """Construct a HandleType tuple matching the
    ``(device, aligned_size, d_mem_ptr, p_memHandle_ptr)`` C-extension ABI."""
    return (device, size, d_mem, p_handle)


def _seed_allocator_with_handles(cumem, allocator, handles):
    """Populate ``allocator.pointer_to_data`` with ``AllocationData`` entries
    that look like a post-sleep state ready to be woken."""
    from vllm.device_allocator import AllocationData

    for handle in handles:
        ptr = handle[2]
        allocator.pointer_to_data[ptr] = AllocationData(
            handle=handle,
            tag="weights",
            cpu_backup_tensor=None,
        )


# -----------------------------------------------------------------------------
# Test 1: per-allocation failures don't abort the loop and don't corrupt state.
# -----------------------------------------------------------------------------


def test_wake_up_propagates_per_allocation_failure_without_corrupting_state(
    cumem_module,
):
    """A failure on the 2nd of 3 allocations must (a) be captured rather
    than silently abort, (b) NOT skip the 3rd allocation, and (c) leave
    ``pointer_to_data`` intact so a retry can re-attempt only the failed
    entry. Pre-fix: the first raise from ``create_and_map`` propagates and
    the 3rd entry never gets mapped, so its restore is skipped silently."""

    allocator = cumem_module.CuMemAllocator.get_instance()
    handles = [
        _make_handle(0, 4096, 0x1000, 0xA000),
        _make_handle(0, 4096, 0x2000, 0xA100),
        _make_handle(0, 4096, 0x3000, 0xA200),
    ]
    _seed_allocator_with_handles(cumem_module, allocator, handles)

    call_log: list[int] = []

    def fake_create_and_map(handle):
        call_log.append(handle[2])
        if handle[2] == 0x2000:
            raise RuntimeError(
                "CUDA Error: invalid argument at csrc/cumem_allocator.cpp:169"
            )

    with mock.patch.object(
        cumem_module, "create_and_map", side_effect=fake_create_and_map
    ):
        with pytest.raises(cumem_module.WakeUpPartialFailure) as excinfo:
            allocator.wake_up()

    # 1. The loop visited ALL three entries — failures don't abort iteration.
    assert call_log == [0x1000, 0x2000, 0x3000]

    # 2. The structured exception lists exactly the failed pointer(s).
    assert excinfo.value.failed_pointers == [0x2000]

    # 3. State is intact: the entry still lives in pointer_to_data so the
    # caller can retry just that one (or fall back to a cold restart).
    assert 0x2000 in allocator.pointer_to_data
    assert len(allocator.pointer_to_data) == 3


# -----------------------------------------------------------------------------
# Test 2: structured exception type, not bare RuntimeError.
# -----------------------------------------------------------------------------


def test_wake_up_raises_structured_exception_on_persistent_failure(cumem_module):
    """When ``create_and_map`` always fails, ``wake_up`` must raise the
    structured ``WakeUpPartialFailure`` (a ``RuntimeError`` subclass)
    carrying the list of failed pointers and the first underlying
    exception. Pre-fix: a generic ``RuntimeError`` from the C extension is
    re-raised verbatim with no structured payload."""

    allocator = cumem_module.CuMemAllocator.get_instance()
    handles = [
        _make_handle(0, 4096, 0x1000, 0xA000),
        _make_handle(0, 4096, 0x2000, 0xA100),
    ]
    _seed_allocator_with_handles(cumem_module, allocator, handles)

    original = RuntimeError(
        "CUDA Error: invalid argument at csrc/cumem_allocator.cpp:169"
    )

    def fake_create_and_map(handle):
        raise original

    with mock.patch.object(
        cumem_module, "create_and_map", side_effect=fake_create_and_map
    ):
        with pytest.raises(cumem_module.WakeUpPartialFailure) as excinfo:
            allocator.wake_up()

    err = excinfo.value
    # Subclass relationship lets existing ``except RuntimeError`` callers
    # still catch it, but new callers can `isinstance(e, WakeUpPartialFailure)`
    # for structured handling.
    assert isinstance(err, RuntimeError)
    assert isinstance(err, cumem_module.WakeUpPartialFailure)
    assert set(err.failed_pointers) == {0x1000, 0x2000}
    assert err.first_exception is original


# -----------------------------------------------------------------------------
# Test 3: structured failure includes ALL failed pointers, not just the first.
# -----------------------------------------------------------------------------


def test_wake_up_collects_all_failed_pointers(cumem_module):
    """The structured exception must record every failed pointer the loop
    encountered, not just the first one. This is what lets a calling
    executor decide between per-allocation retry and a worker-wide cold
    restart based on the breadth of the failure."""

    allocator = cumem_module.CuMemAllocator.get_instance()
    handles = [
        _make_handle(0, 4096, 0x1000, 0xA000),
        _make_handle(0, 4096, 0x2000, 0xA100),
        _make_handle(0, 4096, 0x3000, 0xA200),
        _make_handle(0, 4096, 0x4000, 0xA300),
    ]
    _seed_allocator_with_handles(cumem_module, allocator, handles)

    def fake_create_and_map(handle):
        # Fail on the middle two — succeed on the outer two.
        if handle[2] in (0x2000, 0x3000):
            raise RuntimeError("CUDA Error: invalid argument at .../cumem:169")

    with mock.patch.object(
        cumem_module, "create_and_map", side_effect=fake_create_and_map
    ):
        with pytest.raises(cumem_module.WakeUpPartialFailure) as excinfo:
            allocator.wake_up()

    # All failed pointers reported in iteration order.
    assert excinfo.value.failed_pointers == [0x2000, 0x3000]


# -----------------------------------------------------------------------------
# Test 4: success path is unchanged — no exception, no skips.
# -----------------------------------------------------------------------------


def test_wake_up_success_path_unchanged(cumem_module):
    """Sanity check: when every ``create_and_map`` succeeds, ``wake_up``
    must not raise, must visit every entry, and must not leak the new
    ``WakeUpPartialFailure`` type into the success path."""

    allocator = cumem_module.CuMemAllocator.get_instance()
    handles = [
        _make_handle(0, 4096, 0x1000, 0xA000),
        _make_handle(0, 4096, 0x2000, 0xA100),
    ]
    _seed_allocator_with_handles(cumem_module, allocator, handles)

    call_log: list[int] = []

    def fake_create_and_map(handle):
        call_log.append(handle[2])

    with mock.patch.object(
        cumem_module, "create_and_map", side_effect=fake_create_and_map
    ):
        allocator.wake_up()  # must not raise

    assert call_log == [0x1000, 0x2000]
