# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM storage shared by POSIX descriptors, with tensor-owned mappings."""

from __future__ import annotations

import sys
import weakref
from contextlib import ExitStack, suppress
from dataclasses import dataclass

import torch

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    CachedTensorEntry,
    IPCBackend,
    TensorKind,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.cuda_ipc import (
    TensorEntry,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.serial_utils import (
    SharedFileDescriptor,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    storage_bytes,
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
        self.closed = False
        self.fd: SharedFileDescriptor | None = None
        self.driver = get_vmm_driver()
        if not isinstance(self.driver, CudaVmmDriver):
            raise RuntimeError("The VMM weight-cache backend requires CUDA")
        self.driver.ensure_context(device_index)
        granularity = self.driver.granularity(
            device_index, MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        if size <= 0 or (shared_fd is not None and size % granularity):
            raise ValueError("Invalid VMM weight-cache allocation size")
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
        self.shared = VmmStorage(self.fd, self.size)

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
        if self.closed:
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
            self.closed = True

    def __del__(self, _is_finalizing=sys.is_finalizing):
        if not _is_finalizing():
            with suppress(Exception):
                self.close()


@dataclass
class VmmStorage:
    """A shareable allocation imported on demand for a local device."""

    fd: SharedFileDescriptor
    size: int

    def __getstate__(self):
        return self.fd, self.size

    def __setstate__(self, state):
        self.fd, self.size = state

    def tensor(self, device_index: int) -> torch.Tensor:
        with torch.accelerator.device_index(device_index):
            return VmmAllocation(self.size, device_index, self.fd).tensor()


@dataclass
class VmmTensorEntry:
    kind: TensorKind
    storage: VmmStorage
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int

    def rebuild(
        self,
        device_index: int,
        buffers: dict[object, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Rebuild a view, reusing one buffer per storage and local device."""
        if buffers is None:
            buffers = {}
        key = id(self.storage), device_index
        if key not in buffers:
            buffers[key] = self.storage.tensor(device_index)
        buffer = buffers[key]
        return torch.empty(0, dtype=self.dtype, device=buffer.device).set_(
            buffer.untyped_storage(), self.offset, self.shape, self.stride
        )


class VmmBackend(IPCBackend):
    """Move canonical GPU storages to exportable VMM allocations after warmup."""

    def __init__(self):
        self._allocations: weakref.WeakValueDictionary[
            tuple[torch.device, int], VmmAllocation
        ] = weakref.WeakValueDictionary()

    def materialize(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        groups: dict[tuple[torch.device, int], list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in tensors.items():
            if tensor.is_cuda and tensor.untyped_storage().nbytes():
                key = tensor.device, tensor.untyped_storage().data_ptr()
                groups.setdefault(key, []).append((name, tensor))
        # Rebind one storage at a time to avoid retaining a second complete model.
        for (device, _), views in groups.items():
            assert device.index is not None
            with torch.accelerator.device_index(device.index):
                source = storage_bytes(views[0][1].untyped_storage())
                allocation = VmmAllocation(source.numel(), device.index)
                self._allocations[device, allocation.ptr] = allocation
                buffer = allocation.tensor()
                buffer[: source.numel()].copy_(source)
                torch.accelerator.synchronize(device)
                for name, tensor in views:
                    tensors[name] = torch.empty(
                        0, dtype=tensor.dtype, device=device
                    ).set_(
                        buffer.untyped_storage(),
                        tensor.storage_offset(),
                        tensor.shape,
                        tensor.stride(),
                    )
                del tensor
                views.clear()
                del source, buffer, allocation
                torch.accelerator.empty_cache()
        return tensors

    def export_tensor(
        self, tensor: torch.Tensor, kind: TensorKind
    ) -> CachedTensorEntry:
        if not tensor.is_cuda or not tensor.untyped_storage().nbytes():
            return TensorEntry.from_tensor(tensor, kind)
        allocation = self._allocations.get(
            (tensor.device, tensor.untyped_storage().data_ptr())
        )
        if allocation is None:
            raise RuntimeError("Weight storage was not materialized by the VMM backend")
        return VmmTensorEntry(
            kind,
            allocation.shared,
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )
