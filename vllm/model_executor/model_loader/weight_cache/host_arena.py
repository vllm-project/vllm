# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memfd-backed storage for large CPU model tensors."""

import ctypes
import mmap
import os
import threading
import weakref
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister

HOST_ARENA_MIN_BYTES = 1 << 20
_ALIGNMENT = 256


@dataclass(frozen=True)
class HostArenaDescriptor:
    nbytes: int
    pin_required: bool


@dataclass(frozen=True)
class HostTensorDescriptor:
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


class _MappingLifetime:
    def __init__(self, mapping: mmap.mmap, pin_required: bool):
        self.mapping = mapping
        self.pin_required = pin_required
        self.registered = False
        self.closed = False
        self._pin_attempted = False
        self._lock = threading.Lock()
        self.pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        self.nbytes = len(mapping)

    def ensure_pinned(self) -> None:
        if not self.pin_required:
            return
        with self._lock:
            if self.closed:
                raise RuntimeError("The host arena mapping is closed")
            if self._pin_attempted:
                return
            self._pin_attempted = True
            self.registered = host_register(self.pointer, self.nbytes)
            if not self.registered:
                raise RuntimeError(
                    "cudaHostRegister failed for CPU weights that require pinned memory"
                )

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self.registered:
                self.registered = False
                with suppress(Exception):
                    host_unregister(self.pointer)
            with suppress(BufferError):
                self.mapping.close()

    def __del__(self):
        self.close()


def _tensor_view(base: torch.Tensor, descriptor: HostTensorDescriptor) -> torch.Tensor:
    element_size = descriptor.dtype.itemsize
    if (
        descriptor.offset < 0
        or descriptor.nbytes <= 0
        or descriptor.offset + descriptor.nbytes > base.numel()
        or descriptor.nbytes % element_size
        or descriptor.storage_offset < 0
        or any(size < 0 for size in descriptor.shape)
        or any(stride < 0 for stride in descriptor.stride)
    ):
        raise ValueError("Invalid host arena tensor descriptor")
    typed = base.narrow(0, descriptor.offset, descriptor.nbytes).view(descriptor.dtype)
    if descriptor.storage_offset > typed.numel():
        raise ValueError("Host arena tensor storage offset is out of bounds")
    if not any(size == 0 for size in descriptor.shape):
        extent = descriptor.storage_offset + sum(
            (size - 1) * stride
            for size, stride in zip(descriptor.shape, descriptor.stride)
        )
        if extent >= typed.numel():
            raise ValueError("Host arena tensor view is out of bounds")
    return typed.as_strided(
        descriptor.shape, descriptor.stride, descriptor.storage_offset
    )


def _copy_tensor(tensor: torch.Tensor, pin_required: bool) -> torch.Tensor:
    if tensor.is_cuda or not pin_required:
        return tensor.clone()
    result = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, pin_memory=True
    )
    result.copy_(tensor)
    return result


@dataclass
class _TensorGroup:
    storage: torch.UntypedStorage
    tensors: list[torch.Tensor]

    @property
    def nbytes(self) -> int:
        return self.storage.nbytes()

    @property
    def pin_required(self) -> bool:
        return any(tensor.is_pinned() for tensor in self.tensors)


class _ArenaBase:
    _base: torch.Tensor | None
    _owner: np.ndarray | None
    _lifetime: _MappingLifetime

    def _open_mapping(self, mapping: mmap.mmap, pin_required: bool) -> None:
        self._base = None
        self._owner = None
        self._lifetime = _MappingLifetime(mapping, pin_required)
        try:
            self._owner = np.frombuffer(mapping, dtype=np.uint8)
            self._finalizer = weakref.finalize(self._owner, self._lifetime.close)
            self._finalizer.atexit = False  # type: ignore[misc]
            self._base = torch.from_numpy(self._owner)
            self._lifetime.ensure_pinned()
        except Exception:
            self._base = None
            self._owner = None
            raise

    def _get_tensor(
        self, descriptor: HostTensorDescriptor, *, copy: bool = False
    ) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("The host arena is closed")
        self._lifetime.ensure_pinned()
        tensor = _tensor_view(self._base, descriptor)
        return _copy_tensor(tensor, self._lifetime.pin_required) if copy else tensor

    def _release_views(self) -> None:
        self._base = None
        self._owner = None


class HostArenaMapping(_ArenaBase):
    """Engine-side mmap whose lifetime follows its tensor storages."""

    def __init__(self, descriptor: HostArenaDescriptor, fd: int):
        if descriptor.nbytes <= 0 or os.fstat(fd).st_size < descriptor.nbytes:
            raise ValueError("The host arena backing file is too small")
        mapping = mmap.mmap(
            fd,
            descriptor.nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.descriptor = descriptor
        self._open_mapping(mapping, descriptor.pin_required)

    @classmethod
    def from_fds(
        cls, descriptor: HostArenaDescriptor, fds: Iterable[int]
    ) -> "HostArenaMapping":
        fds = list(fds)
        if len(fds) != 1:
            for fd in fds:
                with suppress(OSError):
                    os.close(fd)
            raise ValueError(f"Expected one host arena FD, received {len(fds)}")
        try:
            return cls(descriptor, fds[0])
        finally:
            with suppress(OSError):
                os.close(fds[0])

    def get_tensor(
        self, descriptor: HostTensorDescriptor, *, copy: bool = False
    ) -> torch.Tensor:
        return self._get_tensor(descriptor, copy=copy)

    def close(self) -> None:
        self._release_views()


class HostArena(_ArenaBase):
    """Daemon-side arena that replaces original CPU storages in place."""

    def __init__(self, groups: list[_TensorGroup]):
        self._fd: int | None = None
        self._descriptors: dict[int, HostTensorDescriptor] = {}
        offsets = []
        offset = 0
        for group in groups:
            offsets.append(offset)
            offset += (group.nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        self.descriptor = HostArenaDescriptor(
            nbytes=offset,
            pin_required=any(group.pin_required for group in groups),
        )
        fd = os.memfd_create("vllm_weight_cache", flags=os.MFD_CLOEXEC)
        self._fd = fd
        try:
            os.ftruncate(fd, self.descriptor.nbytes)
            mapping = mmap.mmap(
                fd,
                self.descriptor.nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self._open_mapping(mapping, self.descriptor.pin_required)
            assert self._base is not None
            views: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for group, group_offset in zip(groups, offsets):
                source = torch.empty(0, dtype=torch.uint8).set_(
                    group.storage, 0, (group.nbytes,), (1,)
                )
                target = self._base.narrow(0, group_offset, group.nbytes)
                target.copy_(source)
                for tensor in group.tensors:
                    descriptor = HostTensorDescriptor(
                        offset=group_offset,
                        nbytes=group.nbytes,
                        dtype=tensor.dtype,
                        shape=tuple(tensor.shape),
                        stride=tuple(tensor.stride()),
                        storage_offset=tensor.storage_offset(),
                    )
                    view = _tensor_view(self._base, descriptor)
                    views.append((tensor, tensor.detach(), view))
                    self._descriptors[id(tensor)] = descriptor
            try:
                for tensor, _, view in views:
                    tensor.data = view
            except Exception:
                for tensor, original, _ in views:
                    tensor.data = original
                raise
        except Exception:
            self.close()
            raise

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, *, min_bytes: int = HOST_ARENA_MIN_BYTES
    ) -> "HostArena | None":
        groups: dict[int, _TensorGroup] = {}
        seen_tensors: set[int] = set()
        tensors = (
            tensor
            for named_tensors in (
                model.named_parameters(remove_duplicate=False),
                model.named_buffers(remove_duplicate=False),
            )
            for _, tensor in named_tensors
        )
        for tensor in tensors:
            if id(tensor) in seen_tensors:
                continue
            seen_tensors.add(id(tensor))
            if (
                tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.is_quantized
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                continue
            storage = tensor.untyped_storage()
            if storage.nbytes() < max(min_bytes, 1):
                continue
            key = storage.data_ptr()
            group = groups.get(key)
            if group is None:
                groups[key] = _TensorGroup(storage, [tensor])
            else:
                group.tensors.append(tensor)
        return cls(list(groups.values())) if groups else None

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("The host arena FD is closed")
        return self._fd

    @property
    def nbytes(self) -> int:
        return self.descriptor.nbytes

    @property
    def pin_required(self) -> bool:
        return self.descriptor.pin_required

    def tensor_descriptor(self, tensor: torch.Tensor) -> HostTensorDescriptor | None:
        return self._descriptors.get(id(tensor))

    def close(self) -> None:
        if self._fd is not None:
            with suppress(OSError):
                os.close(self._fd)
            self._fd = None
        self._release_views()

    def __del__(self):
        self.close()
