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
