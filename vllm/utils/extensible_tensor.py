# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Growable GPU byte buffers backed by driver virtual memory management."""

from __future__ import annotations

import ctypes
from contextlib import suppress

import torch

from vllm.logger import init_logger
from vllm.utils.vmm_driver import get_vmm_driver

logger = init_logger(__name__)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# Floor for the commit granule. ROCm reports a 4 KiB allocation granularity
# (CUDA reports 2 MiB), which would put tens of millions of granules in the
# bookkeeping below for a large KV cache. Commit in larger units instead.
_MIN_GRANULARITY = 2 << 20


def granule_size(device_index: int) -> int:
    """Size of the units physical memory is committed in, for `device_index`.

    At least `_MIN_GRANULARITY`, and always a multiple of what the driver
    requires.
    """
    driver_granularity = get_vmm_driver().granularity(device_index)
    return _round_up(max(driver_granularity, _MIN_GRANULARITY), driver_granularity)


class _VirtualBuffer:
    """Own one device VA reservation and the physical chunks mapped into it.

    Physical memory is committed incrementally, at granularity-sized granules,
    via `ensure_committed_range`; granules already mapped by an earlier
    (possibly overlapping) range are skipped, so ranges may abut or overlap
    freely.
    """

    def __init__(
        self, max_bytes: int, device_index: int, shareable: bool = False
    ) -> None:
        self._driver = get_vmm_driver()
        self._driver.ensure_context(device_index)
        self.device_index = device_index
        self._shareable = shareable

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
        """Granule bounds of the maximal contiguous mapped run covering the
        granules `[first, last)`."""
        while first - 1 in self._mapped_granules:
            first -= 1
        while last in self._mapped_granules:
            last += 1
        return first, last

    def _grant_access(self, first: int, last: int) -> None:
        """Grant device access over the whole granule run `[first, last)`.

        Access is granted per contiguous *run* rather than per newly mapped
        chunk: ROCm rejects a set-access range that starts partway into an
        already-mapped region. Re-granting over the earlier part of the run is
        a no-op, and cheap -- the driver call is flat in the range size.
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
        if self._shareable:
            try:
                handle = driver.create(size, self.device_index, shareable=True)
            except RuntimeError as e:
                logger.warning_once(
                    "Failed to allocate shareable (IPC/RDMA-capable) memory "
                    "(%s); falling back to standard allocation. KV transfers "
                    "from this memory may fail.",
                    e,
                )
                self._shareable = False
                handle = driver.create(size, self.device_index)
        else:
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
        """Unmap and release all physical memory, keeping the VA reservation.

        The base pointer (and any tensor views over it) stays valid but
        unbacked; `ensure_committed_range` maps fresh physical pages again.
        """
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
        shareable: bool = False,
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

        torch.cuda.init()

        self._max_num_bytes: int = max_num_bytes
        self._num_segments: int = num_segments
        self._segment_capacity_bytes: int = max_num_bytes // num_segments
        self._buffer: _VirtualBuffer = _VirtualBuffer(
            max_num_bytes, self._device_index, shareable=shareable
        )
        self._bytes_per_segment: int = 0

    @property
    def tensor(self) -> torch.Tensor:
        """Return a uint8 tensor view of the currently committed prefix."""
        if self._num_segments != 1:
            raise ValueError(
                "tensor (a single committed prefix) is only valid for "
                "num_segments=1; use full_view() and index segments explicitly."
            )
        return uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._bytes_per_segment, self._device_index
        )

    def full_view(self) -> torch.Tensor:
        """Return a uint8 tensor view spanning the requested maximum size."""
        return uint8_tensor_from_ptr(
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

    def release_physical(self) -> None:
        """Release all physical memory while keeping the VA reservation.

        Existing tensor views stay pointer-valid but must not be accessed
        until the buffer is committed again; the data is discarded.
        """
        self._buffer.release_physical()
        self._bytes_per_segment = 0

    @property
    def base_ptr(self) -> int:
        return self._buffer.base_ptr

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self._device_index)

    def free(self) -> None:
        self._buffer.free()
        self._bytes_per_segment = 0


class ExtensibleKVCacheBuffers:
    """Grow-only physical backing for the KV cache: one CUDA virtual-memory
    buffer per KV cache tensor, committed as a per-segment prefix of blocks.

    `commit` maps (and zeroes) physical pages for additional blocks while
    keeping every buffer's base pointer, existing data, and the logical views
    built over the full reserved capacity stable.
    """

    def __init__(
        self,
        buffers: list[tuple[ExtensibleTensor, int]],
        num_blocks_capacity: int,
    ) -> None:
        # Each entry is (buffer, bytes_per_block_per_segment).
        self.buffers = buffers
        self.num_blocks_capacity = num_blocks_capacity
        self.num_blocks_committed = 0
        self._num_blocks_to_recommit = 0

    def commit(self, num_blocks: int, defragment: bool = False) -> None:
        """Grow the committed prefix of every buffer to `num_blocks` blocks.

        With `defragment=True`, all previously committed physical chunks are
        released first so each segment's prefix is re-mapped as one physical
        allocation. Existing contents are DISCARDED, so this is only valid
        before real KV data exists (e.g. right after warmup). It is required
        before KV-transfer registration: UCX cannot transfer memory regions
        that span multiple VMM allocation handles.

        A request at or below the committed size returns without touching the
        mapping, so a non-growing call cannot discard live data -- elastic EP
        re-runs warmup, and hence `ensure_blocks`, while KV data is live.
        """
        if num_blocks <= self.num_blocks_committed:
            return
        if defragment and self.num_blocks_committed > 0:
            self._release()
        for buffer, bytes_per_block_per_segment in self.buffers:
            # Zero only the freshly committed blocks; existing ones are left
            # intact.
            buffer.resize_per_segment_(
                num_blocks * bytes_per_block_per_segment, zero_new=True
            )
        self.num_blocks_committed = num_blocks

    def ensure_blocks(self, num_blocks: int) -> None:
        """Commit at least `num_blocks` blocks, capped at the capacity.

        Warmup paths call this before executing batches that write to a prefix
        of real block IDs, which may exceed what has been committed so far.
        """
        self.commit(min(num_blocks, self.num_blocks_capacity))

    def extend(self, num_blocks: int, defragment: bool = False) -> None:
        """Grow the KV cache to `num_blocks` blocks once the memory actually
        available after warmup and CUDA graph capture is known.

        No re-view is needed: the layers already view the full reserved
        capacity and each block stays at a fixed offset within its layout
        segment, so captured graphs stay valid as more pages are mapped under
        the stable base pointer. Newly committed blocks are zeroed.
        """
        self.commit(num_blocks, defragment=defragment)
        logger.info("Extended KV cache to %d blocks.", num_blocks)

    @property
    def physical_bytes(self) -> int:
        return sum(buffer.physical_bytes for buffer, _ in self.buffers)

    def _release(self) -> None:
        for buffer, _ in self.buffers:
            buffer.release_physical()
        self.num_blocks_committed = 0

    def release_physical(self) -> None:
        """Discard all physical memory (sleep), keeping VA and views valid."""
        self._num_blocks_to_recommit = self.num_blocks_committed
        self._release()

    def recommit(self) -> None:
        """Re-commit the pre-release block count with freshly zeroed pages."""
        self.commit(self._num_blocks_to_recommit)

    def free(self) -> None:
        for buffer, _ in self.buffers:
            buffer.free()
        self.buffers = []
        self.num_blocks_committed = 0


class ExtensibleKVCacheBuilder:
    """Accumulates the VMM-backed buffers backing each KV cache tensor.

    `reserve` stands in for the `torch.zeros(size, dtype=int8)` allocation of
    a KV cache tensor, returning a flat view over the full reserved capacity
    that layer views can be built on. `finish` yields the buffers with a
    single block committed, which is all warmup and CUDA graph capture need.
    """

    def __init__(
        self,
        num_blocks_capacity: int,
        device: torch.device,
        shareable: bool = False,
    ) -> None:
        self.num_blocks_capacity = num_blocks_capacity
        self.device = device
        self.shareable = shareable
        self.buffers: list[tuple[ExtensibleTensor, int]] = []

    def reserve(self, size: int, num_segments: int) -> torch.Tensor:
        """Reserve `size` bytes for one KV cache tensor, in `num_segments`
        equal segments each committed as a prefix of blocks."""
        bytes_per_block, remainder = divmod(size, self.num_blocks_capacity)
        assert not remainder, (
            f"KV cache tensor of {size} bytes does not divide evenly into "
            f"{self.num_blocks_capacity} blocks"
        )
        assert bytes_per_block % num_segments == 0
        buffer = ExtensibleTensor(
            max_num_bytes=size,
            device=self.device,
            num_segments=num_segments,
            shareable=self.shareable,
        )
        self.buffers.append((buffer, bytes_per_block // num_segments))
        return buffer.full_view()

    def finish(self) -> ExtensibleKVCacheBuffers:
        buffers = ExtensibleKVCacheBuffers(self.buffers, self.num_blocks_capacity)
        buffers.commit(1)
        return buffers
