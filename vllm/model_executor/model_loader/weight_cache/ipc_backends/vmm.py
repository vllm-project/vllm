# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM storage shared by POSIX descriptors, with tensor-owned mappings."""

from __future__ import annotations

import sys
from contextlib import ExitStack, suppress
from dataclasses import dataclass

import torch

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    IPCBackend,
    StorageEntry,
    StorageRecord,
    storage_bytes,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.cuda_ipc import (
    TensorEntry,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.serial_utils import (
    SharedFileDescriptor,
)
from vllm.utils.vmm_driver import (
    MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    CudaVmmDriver,
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
        self.driver = get_vmm_driver()
        if not isinstance(self.driver, CudaVmmDriver):
            raise RuntimeError("The VMM weight-cache backend requires CUDA")
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
            self.fd = shared_fd or SharedFileDescriptor(
                self.driver.export_fd(self.handle)
            )
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
        return torch.as_tensor(self, device=torch.device("cuda", self.device_index))

    def close(self) -> None:
        if not self.mapped and not self.handle and not self.ptr:
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
    """A shareable allocation imported on demand for a local device."""

    fd: SharedFileDescriptor
    size: int
    nbytes: int

    def rebuild(self, device_index: int) -> StorageRecord:
        with torch.accelerator.device_index(device_index):
            allocation = VmmAllocation(self.size, device_index, self.fd)
            storage = allocation.tensor().untyped_storage()
        return StorageRecord(
            owner=allocation,
            storage=storage,
            nbytes=self.nbytes,
            allocation_nbytes=self.size,
        )


class VmmBackend(IPCBackend):
    """Move canonical GPU storages to exportable VMM allocations after warmup."""

    def materialize(self, storages: list[torch.UntypedStorage]) -> list[StorageRecord]:
        records = super().materialize(storages)
        for index, record in enumerate(records):
            storage = record.storage
            if not storage.is_cuda or not storage.nbytes():
                continue
            device = storage.device
            assert device.index is not None
            with torch.accelerator.device_index(device.index):
                source = storage_bytes(storage)
                allocation = VmmAllocation(record.nbytes, device.index)
                buffer = allocation.tensor()
                buffer[: source.numel()].copy_(source)
                torch.accelerator.synchronize(device)
                records[index] = StorageRecord(
                    owner=allocation,
                    storage=buffer.untyped_storage(),
                    nbytes=record.nbytes,
                    allocation_nbytes=allocation.size,
                )
                storages[index] = records[index].storage
                del source, buffer, storage, record, allocation
                torch.accelerator.empty_cache()
        return records

    def export_storage(self, record: StorageRecord) -> StorageEntry:
        if isinstance(record.owner, VmmAllocation):
            assert record.owner.fd is not None
            return VmmStorage(record.owner.fd, record.owner.size, record.nbytes)
        return TensorEntry.from_storage(record.storage)
