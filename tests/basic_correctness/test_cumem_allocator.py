# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.device_allocator import cumem
from vllm.device_allocator.cumem import CuMemAllocator


def test_cumem_sleep_discards_unselected_tags(monkeypatch):
    allocator = CuMemAllocator()
    allocator.pointer_to_data = {
        111: cumem.AllocationData((0, 8, 111, 0), "weights"),
        222: cumem.AllocationData((0, 16, 222, 0), "kv_cache"),
    }
    released: list[int] = []
    copied: list[tuple[int, int]] = []

    class FakeCudaRT:
        @staticmethod
        def cudaMemcpy(cpu_ptr, gpu_ptr, size_in_bytes):
            copied.append((gpu_ptr, size_in_bytes))

    monkeypatch.setattr(cumem, "libcudart", FakeCudaRT())
    monkeypatch.setattr(cumem, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        cumem,
        "unmap_and_release",
        lambda handle: released.append(handle[2]),
    )
    monkeypatch.setattr(cumem.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cumem.gc, "collect", lambda: None)

    allocator.sleep(offload_tags=("weights",))

    assert set(released) == {111, 222}
    assert copied == [(111, 8)]
    assert allocator.pointer_to_data[111].cpu_backup_tensor is not None
    assert allocator.pointer_to_data[222].cpu_backup_tensor is None


def _patch_common(monkeypatch, released):
    monkeypatch.setattr(cumem, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        cumem, "unmap_and_release", lambda handle: released.append(handle[2])
    )
    monkeypatch.setattr(cumem, "create_and_map", lambda handle: None)
    monkeypatch.setattr(cumem.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cumem.gc, "collect", lambda: None)


def test_cumem_sleep_async_defers_unmap_until_copies_drain(monkeypatch):
    """With cudaMemcpyAsync available, all D2H copies are enqueued on the
    dedicated stream and synced once before any offloaded handle is
    unmapped (discarded handles may unmap earlier)."""
    allocator = CuMemAllocator()
    allocator.pointer_to_data = {
        111: cumem.AllocationData((0, 8, 111, 0), "weights"),
        222: cumem.AllocationData((0, 16, 222, 0), "weights"),
    }
    events: list[tuple] = []

    class FakeStream:
        cuda_stream = 0

        def synchronize(self):
            events.append(("sync",))

    class FakeCudaRT:
        @staticmethod
        def cudaMemcpyAsync(dst, src, size, stream):
            events.append(("copy", src, size))

    class UnmapLog:
        @staticmethod
        def append(ptr):
            events.append(("unmap", ptr))

    monkeypatch.setattr(cumem, "libcudart", FakeCudaRT())
    _patch_common(monkeypatch, released=UnmapLog())
    monkeypatch.setattr(cumem.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(cumem.torch.cuda, "Stream", FakeStream)

    allocator.sleep(offload_tags=("weights",))

    kinds = [e[0] for e in events]
    assert kinds == ["copy", "copy", "sync", "unmap", "unmap"]


def test_cumem_wake_reuses_pinned_backup_across_cycles(monkeypatch):
    """The host backup buffer survives wake and is reused by the next
    sleep, unless disabled via VLLM_CUMEM_PINNED_CACHE=0."""
    allocator = CuMemAllocator()
    data = cumem.AllocationData((0, 8, 111, 0), "weights")
    allocator.pointer_to_data = {111: data}
    released: list[int] = []

    class FakeCudaRT:
        @staticmethod
        def cudaMemcpy(dst, src, size):
            pass

    monkeypatch.setattr(cumem, "libcudart", FakeCudaRT())
    _patch_common(monkeypatch, released)

    monkeypatch.setenv("VLLM_CUMEM_PINNED_CACHE", "1")
    allocator.sleep(offload_tags=("weights",))
    first_backup = data.cpu_backup_tensor
    assert first_backup is not None
    allocator.wake_up()
    assert data.cpu_backup_tensor is None
    assert data.cpu_backup_cache is first_backup

    allocator.sleep(offload_tags=("weights",))
    assert data.cpu_backup_tensor is first_backup
    assert data.cpu_backup_cache is None

    monkeypatch.setenv("VLLM_CUMEM_PINNED_CACHE", "0")
    allocator.wake_up()
    assert data.cpu_backup_tensor is None
    assert data.cpu_backup_cache is None
