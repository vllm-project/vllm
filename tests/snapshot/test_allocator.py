# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest


def test_pluggable_allocator_requires_loaded_library(monkeypatch):
    import vllm.device_allocator.cumem as cumem

    monkeypatch.setattr(cumem, "init_module", lambda *args: None)
    monkeypatch.setattr(cumem, "lib_name", None)

    with pytest.raises(RuntimeError, match="library is not loaded"):
        cumem.get_pluggable_allocator(
            lambda handle: None, lambda ptr: (0, 0, 0, 0)
        )


def test_selective_discard_preserves_other_allocations(monkeypatch):
    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator()
    weights = (0, 1024, 100, 1)
    kv_cache = (0, 2048, 200, 2)
    allocator.pointer_to_data = {
        100: AllocationData(weights, "weights"),
        200: AllocationData(kv_cache, "kv_cache"),
    }
    discarded: list[Any] = []
    monkeypatch.setattr(cumem, "unmap_and_release", discarded.append)
    monkeypatch.setattr(cumem.torch.accelerator, "synchronize", lambda *a: None)
    monkeypatch.setattr(cumem.gc, "collect", lambda: None)
    monkeypatch.setattr(cumem.torch.cuda, "empty_cache", lambda: None)

    allocator.discard("kv_cache")

    assert discarded == [kv_cache]
    assert not allocator.pointer_to_data[100].is_asleep
    assert allocator.pointer_to_data[200].is_asleep


def test_selective_discard_skips_missing_and_asleep_tags(monkeypatch):
    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator()
    weights = AllocationData((0, 1024, 100, 1), "weights")
    kv_cache = AllocationData((0, 2048, 200, 2), "kv_cache", is_asleep=True)
    allocator.pointer_to_data = {100: weights, 200: kv_cache}
    discarded: list[Any] = []
    monkeypatch.setattr(cumem, "unmap_and_release", discarded.append)
    monkeypatch.setattr(cumem.torch.accelerator, "synchronize", lambda *a: None)

    allocator.discard(("weights", "kv_cache", "missing"))

    assert discarded == [weights.handle]
    assert allocator.pointer_to_data[100].is_asleep
    assert allocator.pointer_to_data[200].is_asleep


def test_selective_wake_maps_only_sleeping_allocations(monkeypatch):
    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator()
    weights = AllocationData((0, 1024, 100, 1), "weights")
    kv_cache = AllocationData((0, 2048, 200, 2), "kv_cache", is_asleep=True)
    allocator.pointer_to_data = {100: weights, 200: kv_cache}
    mapped: list[Any] = []
    monkeypatch.setattr(cumem, "create_and_map", mapped.append)
    monkeypatch.setattr(cumem.gc, "collect", lambda: None)
    monkeypatch.setattr(cumem.torch.accelerator, "empty_cache", lambda: None)

    allocator.wake_up(["weights", "kv_cache"])

    assert mapped == [kv_cache.handle]
    assert not kv_cache.is_asleep


def test_wake_retries_host_copy_without_remapping(monkeypatch):
    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData
    from vllm.device_allocator.cumem import CuMemAllocator

    class Backup:
        def data_ptr(self):
            return 300

        def numel(self):
            return 1024

        def element_size(self):
            return 1

    allocator = CuMemAllocator()
    backup = Backup()
    weights = AllocationData(
        (0, 1024, 100, 1),
        "weights",
        cpu_backup_tensor=backup,
        is_asleep=True,
    )
    allocator.pointer_to_data = {100: weights}
    mapped: list[Any] = []
    copy_attempts = 0

    def copy(*args):
        nonlocal copy_attempts
        copy_attempts += 1
        if copy_attempts == 1:
            raise RuntimeError("copy failed")

    monkeypatch.setattr(cumem, "create_and_map", mapped.append)
    monkeypatch.setattr(cumem.libcudart, "cudaMemcpy", copy)
    monkeypatch.setattr(cumem.gc, "collect", lambda: None)
    monkeypatch.setattr(cumem.torch.accelerator, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="copy failed"):
        allocator.wake_up(["weights"])

    assert mapped == [weights.handle]
    assert not weights.is_asleep
    assert weights.cpu_backup_tensor is backup

    allocator.wake_up(["weights"])

    assert mapped == [weights.handle]
    assert copy_attempts == 2
    assert weights.cpu_backup_tensor is None


def test_sleep_backs_up_only_selected_tags_and_wake_releases_backup(monkeypatch):
    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData
    from vllm.device_allocator.cumem import CuMemAllocator

    class Backup:
        def __init__(self, size):
            self.size = size

        def data_ptr(self):
            return 300

        def numel(self):
            return self.size

        def element_size(self):
            return 1

    allocator = CuMemAllocator()
    weights = AllocationData((0, 1024, 100, 1), "weights")
    kv_cache = AllocationData((0, 2048, 200, 2), "kv_cache")
    allocator.pointer_to_data = {100: weights, 200: kv_cache}
    unmapped: list[Any] = []
    mapped: list[Any] = []
    copies: list[Any] = []
    monkeypatch.setattr(cumem.torch, "empty", lambda size, **kwargs: Backup(size))
    monkeypatch.setattr(
        cumem.libcudart, "cudaMemcpy", lambda *args: copies.append(args)
    )
    monkeypatch.setattr(cumem, "unmap_and_release", unmapped.append)
    monkeypatch.setattr(cumem, "create_and_map", mapped.append)
    monkeypatch.setattr(cumem.gc, "collect", lambda: None)
    monkeypatch.setattr(cumem.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cumem.torch.accelerator, "empty_cache", lambda: None)

    allocator.sleep(offload_tags=("weights",))

    assert unmapped == [weights.handle, kv_cache.handle]
    assert weights.is_asleep and kv_cache.is_asleep
    assert weights.cpu_backup_tensor is not None
    assert kv_cache.cpu_backup_tensor is None
    diagnostics = allocator.allocation_diagnostics()["tags"]
    assert diagnostics["weights"]["host_backup_bytes"] == 1024
    assert diagnostics["kv_cache"]["host_backup_bytes"] == 0
    assert copies == [(300, 100, 1024)]

    allocator.wake_up()

    assert mapped == [weights.handle, kv_cache.handle]
    assert not weights.is_asleep and not kv_cache.is_asleep
    assert weights.cpu_backup_tensor is None
    assert copies == [(300, 100, 1024), (100, 300, 1024)]


def test_allocator_diagnostics_distinguishes_va_and_mapping_state():
    from vllm.device_allocator import AllocationData
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator()
    allocator.pointer_to_data = {
        100: AllocationData((0, 1024, 100, 1), "weights"),
        200: AllocationData((0, 2048, 200, 2), "kv_cache", is_asleep=True),
    }

    diagnostics = allocator.allocation_diagnostics()["tags"]

    assert diagnostics["weights"]["virtual_addresses"] == [100]
    assert diagnostics["weights"]["mapped_virtual_addresses"] == [100]
    assert diagnostics["kv_cache"]["logical_bytes"] == 2048
    assert diagnostics["kv_cache"]["virtual_addresses"] == [200]
    assert diagnostics["kv_cache"]["mapped_virtual_addresses"] == []
