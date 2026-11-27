# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA/ROCm VMM storage shared by POSIX descriptors, with owned mappings."""

from __future__ import annotations

import sys
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from typing import cast

import torch

from vllm.platforms import current_platform
from vllm.preload.ipc_backends.base import (
    IPCBackendBase,
    StorageEntry,
    StorageRecord,
    storage_bytes,
)
from vllm.preload.ipc_backends.memfd_storage import materialize_pinned_cpu_storages
from vllm.preload.ipc_backends.serial_utils import SharedFileDescriptor
from vllm.preload.ipc_backends.torch_ipc import export_tensor
from vllm.utils.vmm_driver import (
    MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    VmmDriver,
    get_vmm_driver,
)


class VmmAllocation:
    """Own a mapping; PyTorch's CUDA array interface retains its Python owner."""

    def __init__(
        self,
        size: int,
        device_index: int,
        shared_fd: SharedFileDescriptor | None = None,
    ):
        self.device_index = device_index
        self.ptr = 0
        self.handle = 0
        self.mapped = False
        self.fd: SharedFileDescriptor | None = None
        self.driver: VmmDriver = get_vmm_driver()
        self.driver.ensure_context(device_index)
        granularity = self.driver.granularity(
            device_index, MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        self.size = (size + granularity - 1) // granularity * granularity
        try:
            self.ptr = self.driver.reserve(self.size)
            self.handle = (
                self.driver.create(
                    self.size, device_index, MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
                )
                if shared_fd is None
                else self.driver.import_fd(shared_fd.fileno())
            )
            self.driver.map(self.ptr, self.size, self.handle)
            self.mapped = True
            self.driver.set_access(self.ptr, self.size, device_index)
            if shared_fd is None:
                self.fd = SharedFileDescriptor(self.driver.export_fd(self.handle))
        except BaseException:
            with suppress(BaseException):
                self.close()
            raise

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "shape": (self.size,),
            "typestr": "|u1",
            "data": (self.ptr, False),
            "version": 3,
        }

    def tensor(self) -> torch.Tensor:
        return torch.as_tensor(
            self,
            device=torch.device(current_platform.device_type, self.device_index),
        )

    def close(self) -> None:
        if not self.mapped and not self.handle and not self.ptr:
            self._finish_close()
            return
        with ExitStack() as stack:
            stack.enter_context(torch.accelerator.device_index(self.device_index))
            self.driver.ensure_context(self.device_index)
            stack.callback(self._finish_close)
            stack.callback(self._free_reserved)
            stack.callback(self._release_handle)
            stack.callback(self._unmap)

    def _unmap(self) -> None:
        if self.mapped:
            # External memory has no caching-allocator stream bookkeeping.
            torch.accelerator.synchronize(self.device_index)
            self.driver.unmap(self.ptr, self.size)
            self.mapped = False

    def _release_handle(self) -> None:
        if self.handle:
            self.driver.release(self.handle)
            self.handle = 0

    def _free_reserved(self) -> None:
        if self.ptr and not self.mapped:
            self.driver.free_reserved(self.ptr, self.size)
            self.ptr = 0

    def _finish_close(self) -> None:
        if not self.mapped and not self.handle and not self.ptr:
            self.fd = None

    def __del__(self, _is_finalizing=sys.is_finalizing):
        if not _is_finalizing():
            with suppress(Exception):
                self.close()


@dataclass
class VmmStorage:
    """A logical storage in an allocation imported on demand for a local device."""

    fd: SharedFileDescriptor
    size: int
    byte_offset: int = 0

    def rebuild(self, device_index: int, cache: dict | None = None) -> StorageRecord:
        base = cache.get(self.fd) if cache is not None else None
        if base is None:
            with torch.accelerator.device_index(device_index):
                allocation = VmmAllocation(self.size, device_index, self.fd)
                storage = allocation.tensor().untyped_storage()
            base = StorageRecord(owner=allocation, storage=storage)
            if cache is not None:
                cache[self.fd] = base
        return StorageRecord(
            owner=base.owner,
            storage=base.storage,
            byte_offset=self.byte_offset,
        )


class VmmIPCBackend(IPCBackendBase):
    """Move canonical GPU storages to exportable VMM allocations after warmup."""

    _ALIGNMENT = 512

    def __init__(self, max_allocation_bytes: int = 1 << 30):
        if max_allocation_bytes < 0:
            raise ValueError("max_allocation_bytes must be non-negative")
        self.max_allocation_bytes = max_allocation_bytes

    @classmethod
    def _plan_packs(
        cls, sizes: list[int], max_allocation_bytes: int
    ) -> list[list[int]]:
        packs: list[list[int]] = []
        offsets: list[int] = []
        end = 0
        for size in sizes:
            if not max_allocation_bytes or size > max_allocation_bytes:
                if offsets:
                    packs.append(offsets)
                    offsets = []
                packs.append([0])
                end = 0
                continue
            offset = (end + cls._ALIGNMENT - 1) // cls._ALIGNMENT * cls._ALIGNMENT
            if offsets and offset + size > max_allocation_bytes:
                packs.append(offsets)
                offsets = []
                offset = 0
            offsets.append(offset)
            end = offset + size
        if offsets:
            packs.append(offsets)
        return packs

    def materialize(self, storages: list[torch.UntypedStorage]) -> list[StorageRecord]:
        records = materialize_pinned_cpu_storages(storages)
        groups: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            storage = record.storage
            if not storage.is_cuda or not storage.nbytes():
                continue
            assert storage.device.index is not None
            groups.setdefault(storage.device.index, []).append(index)

        allocations: list[VmmAllocation] = []
        try:
            for device_index, indices in groups.items():
                sizes = [records[index].storage.nbytes() for index in indices]
                position = 0
                for offsets in self._plan_packs(sizes, self.max_allocation_bytes):
                    pack_indices = indices[position : position + len(offsets)]
                    position += len(offsets)
                    sources = [
                        storage_bytes(records[index].storage) for index in pack_indices
                    ]
                    pack_nbytes = offsets[-1] + sources[-1].numel()
                    with torch.accelerator.device_index(device_index):
                        allocation = VmmAllocation(pack_nbytes, device_index)
                        allocations.append(allocation)
                        buffer = allocation.tensor()
                        for source, offset in zip(sources, offsets, strict=True):
                            buffer[offset : offset + source.numel()].copy_(source)
                        torch.accelerator.synchronize(device_index)
                        for index, offset in zip(pack_indices, offsets, strict=True):
                            records[index] = StorageRecord(
                                owner=allocation,
                                storage=buffer.untyped_storage(),
                                byte_offset=offset,
                            )
                            storages[index] = records[index].storage
                        del sources, source, buffer, allocation
                        torch.accelerator.empty_cache()
        except BaseException:
            for allocation in allocations:
                with suppress(BaseException):
                    allocation.close()
            raise
        return records

    def export_storage(self, record: StorageRecord) -> StorageEntry:
        if not record.storage.is_cuda or not record.storage.nbytes():
            return export_tensor(record)

        owner = cast(VmmAllocation, record.owner)
        assert owner.fd is not None
        return VmmStorage(owner.fd, owner.size, record.byte_offset)
