# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share a pinned CPU weight storage through a local Unix socket."""

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
from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister


def _unregister(mapping: mmap.mmap) -> None:
    # The finalizer holds the mmap until the last tensor view releases its storage.
    host_unregister(ctypes.addressof(ctypes.c_char.from_buffer(mapping)))


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
        from vllm.model_executor.model_loader.weight_cache.protocol import (
            WeightCacheUnavailableError,
        )

        mapping = mmap.mmap(self.fd.fileno(), self.nbytes)
        buffer = (ctypes.c_ubyte * self.nbytes).from_buffer(mapping)
        if not host_register(ctypes.addressof(buffer), self.nbytes):
            raise WeightCacheUnavailableError("Could not register pinned CPU weights")
        finalizer = weakref.finalize(buffer, _unregister, mapping)
        finalizer.atexit = False  # type: ignore[misc]
        tensor = torch.frombuffer(buffer, dtype=torch.uint8)
        return StorageRecord(owner=tensor, storage=tensor.untyped_storage())
