# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Growable CUDA byte buffers backed by CUDA virtual memory management."""

from __future__ import annotations

import ctypes
from contextlib import suppress

import torch

_CUDA_SUCCESS = 0
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOCATION_COMP_NONE = 0


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


_CUdeviceptr = ctypes.c_ulonglong
_CUmemHandle = ctypes.c_ulonglong
_CUcontext = ctypes.c_void_p

_libcuda: ctypes.CDLL | None = None


def _find_loaded_library(lib_name: str) -> str | None:
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if lib_name not in line:
                    continue
                start = line.index("/")
                return line[start:].strip()
    except (OSError, ValueError):
        return None
    return None


def _load_libcuda() -> ctypes.CDLL:
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    if path := _find_loaded_library("libcuda"):
        return ctypes.CDLL(path)
    raise RuntimeError(
        "Could not load libcuda. The CUDA driver library is required for "
        "ExtensibleTensor."
    )


def _configure_signatures(lib: ctypes.CDLL) -> None:
    pointer = ctypes.POINTER
    lib.cuGetErrorString.argtypes = [ctypes.c_int, pointer(ctypes.c_char_p)]
    lib.cuCtxGetCurrent.argtypes = [pointer(_CUcontext)]
    lib.cuDevicePrimaryCtxRetain.argtypes = [pointer(_CUcontext), ctypes.c_int]
    lib.cuCtxSetCurrent.argtypes = [_CUcontext]
    lib.cuMemGetAllocationGranularity.argtypes = [
        pointer(ctypes.c_size_t),
        pointer(_CUmemAllocationProp),
        ctypes.c_int,
    ]
    lib.cuMemAddressReserve.argtypes = [
        pointer(_CUdeviceptr),
        ctypes.c_size_t,
        ctypes.c_size_t,
        _CUdeviceptr,
        ctypes.c_ulonglong,
    ]
    lib.cuMemCreate.argtypes = [
        pointer(_CUmemHandle),
        ctypes.c_size_t,
        pointer(_CUmemAllocationProp),
        ctypes.c_ulonglong,
    ]
    lib.cuMemMap.argtypes = [
        _CUdeviceptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _CUmemHandle,
        ctypes.c_ulonglong,
    ]
    lib.cuMemSetAccess.argtypes = [
        _CUdeviceptr,
        ctypes.c_size_t,
        pointer(_CUmemAccessDesc),
        ctypes.c_size_t,
    ]
    lib.cuMemUnmap.argtypes = [_CUdeviceptr, ctypes.c_size_t]
    lib.cuMemRelease.argtypes = [_CUmemHandle]
    lib.cuMemAddressFree.argtypes = [_CUdeviceptr, ctypes.c_size_t]

    for fn in (
        lib.cuGetErrorString,
        lib.cuCtxGetCurrent,
        lib.cuDevicePrimaryCtxRetain,
        lib.cuCtxSetCurrent,
        lib.cuMemGetAllocationGranularity,
        lib.cuMemAddressReserve,
        lib.cuMemCreate,
        lib.cuMemMap,
        lib.cuMemSetAccess,
        lib.cuMemUnmap,
        lib.cuMemRelease,
        lib.cuMemAddressFree,
    ):
        fn.restype = ctypes.c_int


def _cuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        lib = _load_libcuda()
        _configure_signatures(lib)
        _libcuda = lib
    return _libcuda


def _check(result: int) -> None:
    if result == _CUDA_SUCCESS:
        return
    msg = ctypes.c_char_p()
    _cuda().cuGetErrorString(result, ctypes.byref(msg))
    detail = msg.value.decode() if msg.value else "unknown error"
    raise RuntimeError(f"CUDA driver error {result}: {detail}")


def _ensure_context(device_index: int) -> None:
    pctx = _CUcontext()
    _check(_cuda().cuCtxGetCurrent(ctypes.byref(pctx)))
    if pctx.value:
        return
    _check(_cuda().cuDevicePrimaryCtxRetain(ctypes.byref(pctx), device_index))
    _check(_cuda().cuCtxSetCurrent(pctx))


def _make_alloc_prop(device_index: int) -> _CUmemAllocationProp:
    prop = _CUmemAllocationProp()
    prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_index
    prop.allocFlags.compressionType = _CU_MEM_ALLOCATION_COMP_NONE
    return prop


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class _VirtualBuffer:
    """Own one device VA reservation and the physical chunks mapped into it."""

    def __init__(self, max_bytes: int, device_index: int) -> None:
        _ensure_context(device_index)
        self.device_index = device_index

        prop = _make_alloc_prop(device_index)
        granularity = ctypes.c_size_t()
        _check(
            _cuda().cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(prop),
                _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        self.granularity: int = granularity.value
        self.reserved_size: int = _round_up(max(max_bytes, 1), self.granularity)

        dptr = _CUdeviceptr()
        _check(
            _cuda().cuMemAddressReserve(ctypes.byref(dptr), self.reserved_size, 0, 0, 0)
        )
        self.base_ptr: int = dptr.value

        self.committed_bytes: int = 0
        self._handles: list[tuple[int, int]] = []
        self._freed: bool = False

    def ensure_committed(self, nbytes: int) -> None:
        if nbytes > self.reserved_size:
            raise ValueError(
                f"Requested {nbytes} bytes exceeds reserved capacity "
                f"{self.reserved_size}."
            )
        while self.committed_bytes < nbytes:
            delta = _round_up(nbytes - self.committed_bytes, self.granularity)
            chunk = min(delta, self.reserved_size - self.committed_bytes)
            self._map_chunk(chunk)

    def _map_chunk(self, size: int) -> None:
        _ensure_context(self.device_index)
        prop = _make_alloc_prop(self.device_index)

        handle = _CUmemHandle()
        _check(_cuda().cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0))

        addr = self.base_ptr + self.committed_bytes
        try:
            _check(_cuda().cuMemMap(addr, size, 0, handle, 0))
        except RuntimeError:
            _cuda().cuMemRelease(handle)
            raise

        desc = _CUmemAccessDesc()
        desc.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self.device_index
        desc.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(_cuda().cuMemSetAccess(addr, size, ctypes.byref(desc), 1))

        self._handles.append((handle.value, size))
        self.committed_bytes += size

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        _ensure_context(self.device_index)
        if self._handles:
            torch.cuda.synchronize(self.device_index)
        offset = 0
        for handle, size in self._handles:
            _check(_cuda().cuMemUnmap(self.base_ptr + offset, size))
            _check(_cuda().cuMemRelease(handle))
            offset += size
        if self.base_ptr:
            _check(_cuda().cuMemAddressFree(self.base_ptr, self.reserved_size))
        self._handles = []
        self.committed_bytes = 0
        self.base_ptr = 0

    def __del__(self) -> None:
        with suppress(Exception):
            self.free()


_K_DL_CUDA = 2
_K_DL_UINT = 1
_UINT8_BITS = 8


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]

_KEEPALIVE: dict[int, tuple[object, object, object]] = {}
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _uint8_tensor_from_ptr(ptr: int, num_bytes: int, device_index: int) -> torch.Tensor:
    shape_arr = (ctypes.c_int64 * 1)(num_bytes)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(_K_DL_CUDA, device_index)
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(_K_DL_UINT, _UINT8_BITS, 1)
    managed.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    key = ctypes.addressof(managed)

    def _deleter(_managed_ptr: object) -> None:
        _KEEPALIVE.pop(key, None)

    deleter = _DLDeleter(_deleter)
    managed.deleter = deleter
    _KEEPALIVE[key] = (managed, shape_arr, deleter)

    capsule = _PyCapsule_New(ctypes.addressof(managed), b"dltensor", None)
    return torch.from_dlpack(capsule)


class ExtensibleTensor:
    """A 1-D CUDA byte buffer that can grow without moving its base pointer."""

    def __init__(
        self,
        max_num_bytes: int,
        device: torch.device | str | int | None = None,
    ) -> None:
        if max_num_bytes < 0:
            raise ValueError("max_num_bytes must be non-negative.")

        if device is None:
            device = torch.cuda.current_device()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"ExtensibleTensor requires a cuda device, got {dev}.")
        self._device_index: int = (
            dev.index if dev.index is not None else torch.cuda.current_device()
        )

        torch.cuda.init()

        self._max_num_bytes: int = max_num_bytes
        self._buffer: _VirtualBuffer = _VirtualBuffer(max_num_bytes, self._device_index)
        self._num_bytes: int = 0

    @property
    def tensor(self) -> torch.Tensor:
        """Return a uint8 tensor view of the currently committed prefix."""
        return _uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._num_bytes, self._device_index
        )

    def full_view(self) -> torch.Tensor:
        """Return a uint8 tensor view spanning the requested maximum size."""
        return _uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._max_num_bytes, self._device_index
        )

    def resize_(self, num_bytes: int) -> torch.Tensor:
        """Grow the buffer to `num_bytes` and return the committed-prefix view."""
        if num_bytes > self._max_num_bytes:
            raise ValueError(
                f"Requested {num_bytes} bytes exceeds maximum size "
                f"{self._max_num_bytes}."
            )
        if num_bytes < self._num_bytes:
            raise ValueError(
                f"ExtensibleTensor is grow-only: cannot resize from "
                f"{self._num_bytes} to {num_bytes} bytes."
            )
        self._buffer.ensure_committed(num_bytes)
        self._num_bytes = num_bytes
        return self.tensor

    def append(self, num_bytes: int) -> torch.Tensor:
        """Grow by `num_bytes` additional bytes and return the new view."""
        if num_bytes < 0:
            raise ValueError("num_bytes to append must be non-negative.")
        return self.resize_(self._num_bytes + num_bytes)

    @property
    def num_bytes(self) -> int:
        return self._num_bytes

    @property
    def capacity_bytes(self) -> int:
        return self._buffer.reserved_size

    @property
    def base_ptr(self) -> int:
        return self._buffer.base_ptr

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self._device_index)

    def free(self) -> None:
        self._buffer.free()
        self._num_bytes = 0
