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
    """Own one device VA reservation and the physical chunks mapped into it.

    Physical memory is committed incrementally, at granularity-sized granules,
    via `ensure_committed_range`; granules already mapped by an earlier
    (possibly overlapping) range are skipped, so ranges may abut or overlap
    freely.
    """

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

        # Granule indices (VA offset // granularity) that have physical
        # memory mapped.
        self._mapped_granules: set[int] = set()
        # Each entry is (handle, va_offset, size) for one mapped physical chunk.
        self._handles: list[tuple[int, int, int]] = []
        self._freed: bool = False

    @property
    def committed_bytes(self) -> int:
        """Total physically mapped bytes (a multiple of the granularity)."""
        return len(self._mapped_granules) * self.granularity

    def ensure_committed(self, nbytes: int) -> None:
        """Map physical pages so that at least the first `nbytes` are backed."""
        self.ensure_committed_range(0, nbytes)

    def ensure_committed_range(self, start: int, end: int) -> None:
        """Map physical pages so that the byte range `[start, end)` is backed.

        The range is widened outward to granule boundaries; granules mapped by
        earlier calls are skipped, so a granule shared by two requested ranges
        is mapped once.
        """
        if not 0 <= start <= end:
            raise ValueError(f"Invalid range [{start}, {end}).")
        if end > self.reserved_size:
            raise ValueError(
                f"Requested range end {end} exceeds reserved capacity "
                f"{self.reserved_size}."
            )
        if start == end:
            return
        first = start // self.granularity
        last = (end + self.granularity - 1) // self.granularity  # exclusive
        run_start: int | None = None
        for g in range(first, last + 1):
            unmapped = g < last and g not in self._mapped_granules
            if unmapped and run_start is None:
                run_start = g
            elif not unmapped and run_start is not None:
                self._map_chunk_at(
                    run_start * self.granularity, (g - run_start) * self.granularity
                )
                self._mapped_granules.update(range(run_start, g))
                run_start = None

    def _map_chunk_at(self, offset: int, size: int) -> None:
        """Create one physical chunk of `size` bytes and map it at `offset`."""
        _ensure_context(self.device_index)
        prop = _make_alloc_prop(self.device_index)

        handle = _CUmemHandle()
        _check(_cuda().cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0))

        addr = self.base_ptr + offset
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

        self._handles.append((handle.value, offset, size))

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        _ensure_context(self.device_index)
        if self._handles:
            torch.cuda.synchronize(self.device_index)
        for handle, offset, size in self._handles:
            _check(_cuda().cuMemUnmap(self.base_ptr + offset, size))
            _check(_cuda().cuMemRelease(handle))
        if self.base_ptr:
            _check(_cuda().cuMemAddressFree(self.base_ptr, self.reserved_size))
        self._handles = []
        self._mapped_granules = set()
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
    """A 1-D CUDA byte buffer that can grow without moving its base pointer.

    With `num_segments > 1` the reservation is divided into that many equal
    segments that grow in lockstep via `resize_per_segment_`: the committed
    bytes form a prefix of each segment (segment `i` spans
    `[i * segment_capacity_bytes, (i + 1) * segment_capacity_bytes)` of
    `full_view()`). This backs layouts whose block dimension is not outermost,
    e.g. a K/V-split KV cache (`num_segments=2`). `resize_` / `tensor` /
    `append` assume a single contiguous prefix and are only valid when
    `num_segments == 1`.
    """

    def __init__(
        self,
        max_num_bytes: int,
        device: torch.device | str | int | None = None,
        num_segments: int = 1,
    ) -> None:
        if max_num_bytes < 0:
            raise ValueError("max_num_bytes must be non-negative.")
        if num_segments < 1:
            raise ValueError(f"num_segments must be positive, got {num_segments}.")
        if max_num_bytes % num_segments != 0:
            raise ValueError(
                f"max_num_bytes ({max_num_bytes}) must be divisible by "
                f"num_segments ({num_segments})."
            )

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
        self._num_segments: int = num_segments
        self._segment_capacity_bytes: int = max_num_bytes // num_segments
        self._buffer: _VirtualBuffer = _VirtualBuffer(max_num_bytes, self._device_index)
        self._bytes_per_segment: int = 0

    @property
    def tensor(self) -> torch.Tensor:
        """Return a uint8 tensor view of the currently committed prefix."""
        if self._num_segments != 1:
            raise ValueError(
                "tensor (a single committed prefix) is only valid for "
                "num_segments=1; use full_view() and index segments explicitly."
            )
        return _uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._bytes_per_segment, self._device_index
        )

    def full_view(self) -> torch.Tensor:
        """Return a uint8 tensor view spanning the requested maximum size."""
        return _uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._max_num_bytes, self._device_index
        )

    def resize_(self, num_bytes: int) -> torch.Tensor:
        """Grow the buffer to `num_bytes` and return the committed-prefix view."""
        if self._num_segments != 1:
            raise ValueError(
                "resize_ (a single committed prefix) is only valid for "
                "num_segments=1; use resize_per_segment_."
            )
        self.resize_per_segment_(num_bytes)
        return self.tensor

    def resize_per_segment_(
        self, bytes_per_segment: int, zero_new: bool = False
    ) -> None:
        """Grow every segment's committed prefix to `bytes_per_segment` bytes.

        Existing bytes are preserved and the base pointer is unchanged. With
        `zero_new=True` the newly committed byte range of each segment is
        zeroed (bytes committed earlier are left intact). Raises if
        `bytes_per_segment` is smaller than the current per-segment size
        (shrink is unsupported) or larger than `segment_capacity_bytes`.
        """
        old = self._bytes_per_segment
        if bytes_per_segment < old:
            raise ValueError(
                f"ExtensibleTensor is grow-only: cannot resize from {old} "
                f"to {bytes_per_segment} bytes per segment."
            )
        if bytes_per_segment > self._segment_capacity_bytes:
            raise ValueError(
                f"Requested {bytes_per_segment} bytes per segment exceeds the "
                f"segment capacity {self._segment_capacity_bytes}."
            )
        if bytes_per_segment == old:
            return
        for i in range(self._num_segments):
            start = i * self._segment_capacity_bytes
            self._buffer.ensure_committed_range(start + old, start + bytes_per_segment)
        self._bytes_per_segment = bytes_per_segment
        if zero_new:
            full = self.full_view()
            for i in range(self._num_segments):
                start = i * self._segment_capacity_bytes
                full[start + old : start + bytes_per_segment].zero_()

    def append(self, num_bytes: int) -> torch.Tensor:
        """Grow by `num_bytes` additional bytes and return the new view."""
        if num_bytes < 0:
            raise ValueError("num_bytes to append must be non-negative.")
        return self.resize_(self._bytes_per_segment + num_bytes)

    @property
    def num_bytes(self) -> int:
        """Current committed size in bytes, summed over all segments."""
        return self._bytes_per_segment * self._num_segments

    @property
    def bytes_per_segment(self) -> int:
        """Current committed prefix size of each segment in bytes."""
        return self._bytes_per_segment

    @property
    def num_segments(self) -> int:
        """Number of equal segments the reservation is divided into."""
        return self._num_segments

    @property
    def segment_capacity_bytes(self) -> int:
        """Maximum size of each segment (`max_num_bytes / num_segments`)."""
        return self._segment_capacity_bytes

    @property
    def capacity_bytes(self) -> int:
        return self._buffer.reserved_size

    @property
    def physical_bytes(self) -> int:
        """Physically mapped bytes (committed size rounded up to granules)."""
        return self._buffer.committed_bytes

    @property
    def base_ptr(self) -> int:
        return self._buffer.base_ptr

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self._device_index)

    def free(self) -> None:
        self._buffer.free()
        self._bytes_per_segment = 0
