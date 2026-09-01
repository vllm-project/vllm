# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Growable device byte buffers backed by GPU virtual memory management."""

from __future__ import annotations

import ctypes
from contextlib import suppress

import torch

from vllm.utils.vmm_driver import get_vmm_driver


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ROCm reports a 4 KiB granularity (CUDA 2 MiB); commit in larger units to
# keep the granule bookkeeping small.
_MIN_GRANULARITY = 2 << 20


def granule_size(device_index: int) -> int:
    """Commit granule for ``device_index``: driver granularity, floored at 2 MiB."""
    driver_granularity = get_vmm_driver().granularity(device_index)
    return _round_up(max(driver_granularity, _MIN_GRANULARITY), driver_granularity)


class _VirtualBuffer:
    """One device VA reservation and the physical chunks mapped into it.

    Memory is committed in granules via `ensure_committed_range`; already-mapped
    granules are skipped, so ranges may abut or overlap.
    """

    def __init__(self, max_bytes: int, device_index: int) -> None:
        self._driver = get_vmm_driver()
        self._driver.ensure_context(device_index)
        self.device_index = device_index

        self.granularity: int = granule_size(device_index)
        self.reserved_size: int = _round_up(max(max_bytes, 1), self.granularity)
        self.base_ptr: int = self._driver.reserve(self.reserved_size)

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

    def ensure_committed_range(self, start: int, end: int) -> None:
        """Map physical pages so that the byte range `[start, end)` is backed."""
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
        mapped: list[tuple[int, int]] = []
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
                mapped.append((run_start, g))
                run_start = None

        for chunk_first, chunk_last in mapped:
            self._grant_access(*self._run_bounds(chunk_first, chunk_last))

    def _run_bounds(self, first: int, last: int) -> tuple[int, int]:
        """Bounds of the maximal contiguous mapped run covering `[first, last)`."""
        while first - 1 in self._mapped_granules:
            first -= 1
        while last in self._mapped_granules:
            last += 1
        return first, last

    def _grant_access(self, first: int, last: int) -> None:
        """Grant device access over the granule run `[first, last)`.

        Per run rather than per chunk: ROCm rejects a set-access range that starts
        inside an already-mapped region. Re-granting is a cheap no-op.
        """
        self._driver.set_access(
            self.base_ptr + first * self.granularity,
            (last - first) * self.granularity,
            self.device_index,
        )

    def _map_chunk_at(self, offset: int, size: int) -> None:
        """Create one physical chunk of `size` bytes and map it at `offset`."""
        driver = self._driver
        driver.ensure_context(self.device_index)
        try:
            handle = driver.create(size, self.device_index)
        except RuntimeError:
            # The VMM allocator cannot reuse memory idling in torch's
            # caching allocator; return it to the driver and retry once.
            torch.accelerator.empty_cache()
            handle = driver.create(size, self.device_index)

        addr = self.base_ptr + offset
        try:
            driver.map(addr, size, handle)
        except RuntimeError:
            driver.release(handle)
            raise
        # Access is granted by the caller, per contiguous run.
        self._handles.append((handle, offset, size))

    def release_physical(self) -> None:
        """Unmap and release all physical memory, keeping the VA reservation."""
        driver = self._driver
        driver.ensure_context(self.device_index)
        if self._handles:
            torch.accelerator.synchronize(self.device_index)
        for handle, offset, size in self._handles:
            driver.unmap(self.base_ptr + offset, size)
            driver.release(handle)
        self._handles = []
        self._mapped_granules = set()

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        self.release_physical()
        if self.base_ptr:
            self._driver.free_reserved(self.base_ptr, self.reserved_size)
        self.base_ptr = 0

    def __del__(self) -> None:
        with suppress(Exception):
            self.free()


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


def uint8_tensor_from_ptr(ptr: int, num_bytes: int, device_index: int) -> torch.Tensor:
    """Wrap device memory at `ptr` as a uint8 tensor; the caller keeps it mapped."""
    shape_arr = (ctypes.c_int64 * 1)(num_bytes)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    device_type = get_vmm_driver().dlpack_device_type
    managed.dl_tensor.device = _DLDevice(device_type, device_index)
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
    """A device byte buffer that grows without moving its base pointer.

    The reservation is `num_segments` equal segments that grow in lockstep via
    `resize_per_segment_`; the committed bytes form a prefix of each segment.
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
            device = torch.accelerator.current_device_index()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"ExtensibleTensor requires a cuda device, got {dev}.")
        self._device_index: int = (
            dev.index
            if dev.index is not None
            else torch.accelerator.current_device_index()
        )

        self._max_num_bytes: int = max_num_bytes
        self._num_segments: int = num_segments
        self._segment_capacity_bytes: int = max_num_bytes // num_segments
        self._buffer: _VirtualBuffer = _VirtualBuffer(max_num_bytes, self._device_index)
        self._bytes_per_segment: int = 0

    def full_view(self) -> torch.Tensor:
        """Uint8 view spanning the whole reservation."""
        return uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._max_num_bytes, self._device_index
        )

    def segment_view(self, index: int) -> torch.Tensor:
        """Uint8 tensor over one segment's committed prefix, sized exactly to it."""
        if not 0 <= index < self._num_segments:
            raise IndexError(f"Segment {index} out of range ({self._num_segments}).")
        return uint8_tensor_from_ptr(
            self._buffer.base_ptr + index * self._segment_capacity_bytes,
            self._bytes_per_segment,
            self._device_index,
        )

    def resize_per_segment_(
        self, bytes_per_segment: int, zero_new: bool = False
    ) -> None:
        """Grow every segment's committed prefix; optionally zero the new bytes."""
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

    def release_physical(self) -> None:
        """Drop all physical pages but keep the reservation and its views."""
        self._buffer.release_physical()
        self._bytes_per_segment = 0

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
        """Physically mapped bytes, including granule rounding."""
        return self._buffer.committed_bytes

    @property
    def granularity(self) -> int:
        return self._buffer.granularity

    @property
    def base_ptr(self) -> int:
        return self._buffer.base_ptr

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self._device_index)

    def free(self) -> None:
        self._buffer.free()
        self._bytes_per_segment = 0
