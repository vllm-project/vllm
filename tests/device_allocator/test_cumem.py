# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CuMemAllocator.wake_up()'s ordering contract.

These construct a CuMemAllocator directly (not via the CUDA-only singleton)
and monkeypatch its native-call boundary, so they run without a GPU and
without touching allocation size/insertion order to prove correctness --
see https://github.com/vllm-project/vllm/issues/50011.
"""

import torch

from vllm.device_allocator import AllocationData
from vllm.device_allocator.cumem import CuMemAllocator


def _make_backup_tensor(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, device="cpu")


def _add_entry(
    allocator: CuMemAllocator,
    ptr: int,
    tag: str,
    *,
    backed_up: bool,
    size: int = 16,
) -> AllocationData:
    handle = (0, size, ptr, ptr)  # (device, size, ptr, native handle)
    data = AllocationData(
        handle=handle,
        tag=tag,
        cpu_backup_tensor=_make_backup_tensor(size) if backed_up else None,
        is_asleep=True,
    )
    allocator.pointer_to_data[ptr] = data
    return data


def _patch_native_calls(allocator, monkeypatch):
    """Record create_and_map / cudaMemcpy / empty_host_cache calls in order."""
    import vllm.device_allocator.cumem as cumem_mod

    calls: list[tuple] = []

    def fake_create_and_map(handle):
        calls.append(("map", handle[2]))

    class FakeLibcudart:
        def cudaMemcpy(self, dst, src, count):
            calls.append(("restore", dst))

    def fake_empty_host_cache():
        calls.append(("empty_host_cache",))

    monkeypatch.setattr(cumem_mod, "create_and_map", fake_create_and_map)
    monkeypatch.setattr(cumem_mod, "libcudart", FakeLibcudart())
    monkeypatch.setattr(torch.accelerator, "empty_host_cache", fake_empty_host_cache)
    # gc.collect()/empty_cache() at the top of wake_up() are harmless no-ops
    # here; leave them untouched so we exercise the real function.
    return calls


def test_wake_up_restores_before_flush_before_discarded_remap(monkeypatch):
    """Interleaved insertion order must not affect operation order: all
    backup restores happen before the single host-cache flush, which
    happens before any discarded (no-backup) remap. Fails under the old
    implementation, which never flushed the host cache at all."""
    allocator = CuMemAllocator()
    calls = _patch_native_calls(allocator, monkeypatch)

    # Intentionally interleaved: backup, discard, backup, discard.
    _add_entry(allocator, ptr=100, tag="weights", backed_up=True)
    _add_entry(allocator, ptr=200, tag="kv_cache", backed_up=False)
    _add_entry(allocator, ptr=300, tag="weights", backed_up=True)
    _add_entry(allocator, ptr=400, tag="kv_cache", backed_up=False)

    allocator.wake_up()

    restore_idxs = [i for i, c in enumerate(calls) if c[0] == "restore"]
    flush_idxs = [i for i, c in enumerate(calls) if c[0] == "empty_host_cache"]
    discarded_map_idxs = [
        i for i, c in enumerate(calls) if c[0] == "map" and c[1] in (200, 400)
    ]

    assert len(flush_idxs) == 1
    assert max(restore_idxs) < flush_idxs[0] < min(discarded_map_idxs)
    # Each backed-up entry is mapped immediately before its own restore.
    assert calls.index(("map", 100)) < calls.index(("restore", 100))
    assert calls.index(("map", 300)) < calls.index(("restore", 300))


def test_wake_up_tag_filtering_and_flush_semantics(monkeypatch):
    """Tag-scoped wakes only touch matching entries, and the host cache is
    flushed exactly once when a backup is restored -- never for a KV-only
    wake that restores no backups."""
    allocator = CuMemAllocator()
    calls = _patch_native_calls(allocator, monkeypatch)

    weights = _add_entry(allocator, ptr=100, tag="weights", backed_up=True)
    kv = _add_entry(allocator, ptr=200, tag="kv_cache", backed_up=False)
    other = _add_entry(allocator, ptr=300, tag="other", backed_up=True)

    # KV-only wake: no backups restored, so no flush; other tags untouched.
    allocator.wake_up(tags=["kv_cache"])
    assert kv.is_asleep is False
    assert not any(c[0] == "empty_host_cache" for c in calls)
    assert weights.is_asleep is True
    assert other.is_asleep is True

    # Weights-only wake: restores and flushes exactly once; "other" tag
    # (also backed up) is left untouched despite sharing that property.
    calls.clear()
    allocator.wake_up(tags=["weights"])
    assert weights.is_asleep is False
    assert weights.cpu_backup_tensor is None
    assert sum(1 for c in calls if c[0] == "empty_host_cache") == 1
    assert other.is_asleep is True
    assert other.cpu_backup_tensor is not None
