# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.device_allocator.cumem as cumem
from vllm.device_allocator import AllocationData

pytestmark = pytest.mark.cpu_test


def test_free_callback_synchronizes_handle_device(monkeypatch):
    """The free path syncs the allocation's device before returning its handle."""
    allocator = cumem.CuMemAllocator()
    handle = (1, 1 << 20, 0xC000, 0)
    allocator.pointer_to_data[handle[2]] = AllocationData(handle, "weights")

    sync_calls = []

    def synchronize(device_index=None):
        sync_calls.append(device_index)

    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: pytest.fail(
            "_python_free_callback must not use the legacy CUDA synchronize API"
        ),
    )

    returned = allocator._python_free_callback(handle[2])

    assert sync_calls == [1]
    assert returned is handle
    assert handle[2] not in allocator.pointer_to_data


def test_free_callback_skips_sync_for_asleep_rocm_allocation(monkeypatch):
    """An already-unmapped ROCm allocation returns no chunks without syncing."""
    allocator = cumem.CuMemAllocator()
    handle = (2, 1 << 20, 0xD000, [7, 8])
    allocator.pointer_to_data[handle[2]] = AllocationData(
        handle, "weights", is_asleep=True
    )
    monkeypatch.setattr(
        cumem, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda *args, **kwargs: pytest.fail(
            "an already-unmapped allocation must not synchronize"
        ),
    )

    returned = allocator._python_free_callback(handle[2])

    assert returned == (2, 1 << 20, 0xD000, [])
    assert handle[2] not in allocator.pointer_to_data


def test_sleep_empties_accelerator_cache_after_gc(monkeypatch):
    """sleep() frees the accelerator cache once, after gc and its bookkeeping."""
    allocator = cumem.CuMemAllocator()
    events = []

    monkeypatch.setattr(cumem.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: events.append("empty_cache")
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("sleep() must not use the legacy CUDA cache API"),
    )

    allocator.sleep(offload_tags=())

    assert events == ["gc", "empty_cache"]
    assert allocator.pointer_to_data == {}
    assert allocator.get_current_usage() == 0
