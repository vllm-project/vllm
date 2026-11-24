# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memfd-backed storage entries for shared pinned CPU weights."""

from __future__ import annotations

import ctypes
import mmap
import os
import weakref
from dataclasses import dataclass

import torch

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    StorageRecord,
    storage_bytes,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.serial_utils import (
    SharedFileDescriptor,
)
from vllm.v1.kv_offload.cpu.platforms.base import DeviceRuntime
from vllm.v1.kv_offload.cpu.platforms.factory import get_device_runtime


def _unregister(runtime: DeviceRuntime, mapping: mmap.mmap) -> None:
    # The finalizer holds the mmap until the last tensor view releases its storage.
    runtime.unregister_host(ctypes.addressof(ctypes.c_char.from_buffer(mapping)))


@dataclass(slots=True)
class MemfdStorageEntry:
    fd: SharedFileDescriptor
    nbytes: int

    @classmethod
    def from_storage(cls, source: torch.UntypedStorage) -> StorageRecord:
        size = source.nbytes()
        fd = SharedFileDescriptor(os.memfd_create("vllm_weight_cache", os.MFD_CLOEXEC))
        os.ftruncate(fd.fileno(), size)
        mapping = mmap.mmap(fd.fileno(), size)
        tensor = torch.frombuffer(mapping, dtype=torch.uint8)
        tensor.copy_(storage_bytes(source))
        return StorageRecord(owner=cls(fd, size), storage=tensor.untyped_storage())

    def rebuild(self, device_index: int, cache: dict | None = None) -> StorageRecord:
        mapping = mmap.mmap(self.fd.fileno(), self.nbytes)
        buffer = (ctypes.c_ubyte * self.nbytes).from_buffer(mapping)
        runtime = get_device_runtime()
        if not runtime.register_host(ctypes.addressof(buffer), self.nbytes):
            del buffer
            mapping.close()
            raise ValueError("Could not register pinned CPU weights")
        finalizer = weakref.finalize(buffer, _unregister, runtime, mapping)
        finalizer.atexit = False  # type: ignore[misc]
        tensor = torch.frombuffer(buffer, dtype=torch.uint8)
        return StorageRecord(owner=tensor, storage=tensor.untyped_storage())


def materialize_pinned_cpu_storages(
    storages: list[torch.UntypedStorage],
) -> list[StorageRecord]:
    """Move nonempty pinned CPU storages into memfds and wrap the other storages."""
    records = []
    for index in range(len(storages)):
        storage = storages[index]
        if storage.device.type == "cpu" and storage.nbytes() and storage.is_pinned():
            record = MemfdStorageEntry.from_storage(storage)
            # Release the old pinned allocation before copying the next table.
            storages[index] = record.storage
            del storage
            torch.accelerator.empty_host_cache()
        else:
            record = StorageRecord(owner=storage, storage=storage)
        records.append(record)
    return records
