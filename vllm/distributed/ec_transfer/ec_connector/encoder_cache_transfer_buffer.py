# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple buffer manager for Encoder Cache transfer.

This module provides a simple memory buffer manager for storing and
retrieving encoder cache tensors during distributed transfer operations.

Allocation strategy: circular (ring) buffer.

  - _next_offset is the write head. It advances forward on every
    allocation and wraps to 0 when the remaining tail is too small.
  - Pinned slots are treated as in-flight transfer memory and are never
    evicted or reused.
  - Before placing a new slot we evict every unpinned live slot whose
    region overlaps the chosen allocation range. The slots the write
    head hits first are the oldest ones, giving natural LRU behaviour
    without any explicit free-list or coalescing step.
  - An explicit free() (e.g. on transfer failure) simply removes the
    slot from the tracking structures; the reclaimed gap will be
    silently overwritten the next time the write head passes over it.
"""

import bisect
import ctypes
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BufferSlot:
    """Represents an allocated slot in the buffer."""

    offset: int  # Offset from base address in bytes
    size: int  # Size in bytes


class EncoderCacheTransferBuffer:
    """Ring-buffer manager for encoder cache transfer.

    Manages a contiguous memory region using a circular allocation
    strategy with implicit LRU eviction.

    Supports both GPU (CUDA) and CPU (pinned) memory for RDMA transfers.

    Args:
        buffer_size: Total size of the buffer in bytes.
        device: Device to allocate the buffer on ("cuda" or "cpu").
            For CPU, pinned memory is used for RDMA compatibility.

    Attributes:
        buffer_size: Total buffer size in bytes.
        base_tensor: The underlying tensor.
        base_address: Base memory address of the buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        device: str = "cuda",
    ):
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")

        self.buffer_size = buffer_size
        self.device = device

        # Allocate byte-addressed contiguous memory so transfer_buffer_size
        # exactly matches the registered backing storage capacity.
        if device == "cpu":
            self.base_tensor = torch.empty(
                buffer_size, dtype=torch.uint8, pin_memory=True
            )
        else:
            self.base_tensor = torch.empty(
                buffer_size, dtype=torch.uint8, device=device
            ).contiguous()
        self.base_address = self.base_tensor.data_ptr()

        # addr -> BufferSlot for all live allocations
        self._allocated: dict[int, BufferSlot] = {}
        # addr -> active pin count. Pinned slots cannot be evicted or freed.
        self._pin_counts: dict[int, int] = {}

        # Parallel sorted list of (offset, addr) used for O(log n) range
        # queries when deciding what to evict.  Always kept in sync with
        # self._allocated.
        self._offset_index: list[tuple[int, int]] = []  # sorted by offset

        self._lock = threading.Lock()

        # Ring-buffer write head: next allocation starts here.
        self._next_offset: int = 0

        # Optional callback fired whenever a slot is evicted or freed.
        self.on_free: Callable[[int], None] | None = None

        logger.debug(
            "EncoderCacheTransferBuffer initialized: size=%d bytes, "
            "base_addr=0x%x, device=%s",
            buffer_size,
            self.base_address,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, size: int) -> int:
        """Allocate a slot of the given size using the ring-buffer strategy.

        Args:
            size: Size to allocate in bytes.

        Returns:
            Address of the allocated slot.

        Raises:
            ValueError: If size exceeds buffer capacity.
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")
        if size > self.buffer_size:
            raise ValueError(
                f"Requested size {size} exceeds buffer capacity {self.buffer_size}"
            )

        evicted_addrs: list[int]
        with self._lock:
            offset = self._find_alloc_offset(size)
            if offset is None:
                raise BufferError(
                    f"No unpinned EC transfer buffer space available for {size} bytes"
                )

            # Evict all live slots whose region overlaps [offset, offset+size).
            evicted_addrs = self._evict_range(offset, offset + size)

            # Place the new slot.
            addr = self.base_address + offset
            self._allocated[addr] = BufferSlot(offset=offset, size=size)
            bisect.insort(self._offset_index, (offset, addr))
            self._next_offset = offset + size

            logger.debug(
                "Allocated slot: addr=0x%x, offset=%d, size=%d, next_offset=%d",
                addr,
                offset,
                size,
                self._next_offset,
            )

        self._notify_free(evicted_addrs)
        return addr

    def free(self, addr: int) -> None:
        """Explicitly free an allocated slot (e.g. on transfer failure).

        The reclaimed region is left as a silent gap; it will be
        overwritten the next time the ring write head passes over it.

        Args:
            addr: Address of the slot to free.

        Raises:
            ValueError: If address is not allocated.
        """
        should_notify = False
        with self._lock:
            if addr not in self._allocated:
                raise ValueError(f"Address 0x{addr:x} is not allocated")
            if self._pin_counts.get(addr, 0) > 0:
                raise ValueError(f"Address 0x{addr:x} is pinned")

            slot = self._allocated.pop(addr)
            self._remove_from_index(slot.offset, addr)
            logger.debug("Freed slot: addr=0x%x, size=%d", addr, slot.size)
            should_notify = True

        if should_notify:
            self._notify_free([addr])

    def pin(self, addr: int) -> None:
        """Pin an allocated slot so it cannot be evicted or freed."""
        with self._lock:
            if addr not in self._allocated:
                raise ValueError(f"Address 0x{addr:x} is not allocated")
            self._pin_counts[addr] = self._pin_counts.get(addr, 0) + 1
            logger.debug(
                "Pinned slot: addr=0x%x, pin_count=%d",
                addr,
                self._pin_counts[addr],
            )

    def unpin(self, addr: int) -> None:
        """Release one pin reference from an allocated slot."""
        with self._lock:
            pin_count = self._pin_counts.get(addr, 0)
            if pin_count == 0:
                raise ValueError(f"Address 0x{addr:x} is not pinned")
            if pin_count == 1:
                self._pin_counts.pop(addr)
            else:
                self._pin_counts[addr] = pin_count - 1
            logger.debug(
                "Unpinned slot: addr=0x%x, pin_count=%d",
                addr,
                self._pin_counts.get(addr, 0),
            )

    def is_pinned(self, addr: int) -> bool:
        """Return whether an allocated slot has active pins."""
        with self._lock:
            return self._pin_counts.get(addr, 0) > 0

    def store_tensor(self, tensor: torch.Tensor) -> int:
        """Store a tensor in the buffer.

        Args:
            tensor: Tensor to store.

        Returns:
            Address where the tensor is stored.

        Raises:
            ValueError: If tensor device is incompatible or allocation fails.
        """
        if self.device != "cpu" and not tensor.is_cuda:
            raise ValueError(
                f"GPU buffer requires CUDA tensor, got tensor on {tensor.device}"
            )

        size = tensor.numel() * tensor.element_size()
        addr = self.allocate(size)

        with self._lock:
            slot = self._allocated.get(addr)
            if slot is None:
                raise ValueError("Allocation failed")

        try:
            buffer = (ctypes.c_byte * slot.size).from_address(addr)
            buffer_tensor = torch.frombuffer(
                buffer, dtype=tensor.dtype, count=tensor.numel()
            ).reshape(tensor.shape)
            buffer_tensor.copy_(tensor)
        except Exception as e:
            self.free(addr)
            raise ValueError(f"Failed to store tensor: {e}") from e

        return addr

    def load_tensor(
        self,
        addr: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        copy: bool = True,
    ) -> torch.Tensor:
        """Load a tensor from the buffer.

        Args:
            addr: Address where tensor is stored.
            dtype: Data type of the tensor.
            shape: Shape of the tensor.
            device: Target device for the loaded tensor.
            copy: If True, copy to device. If False, return a view.

        Returns:
            The loaded tensor.

        Raises:
            ValueError: If address is invalid or parameters don't match.
        """
        with self._lock:
            if addr not in self._allocated:
                raise ValueError(f"Address 0x{addr:x} is not allocated")
            slot = self._allocated[addr]

        num_elements = 1
        for dim in shape:
            num_elements *= dim

        element_size = torch.tensor([], dtype=dtype).element_size()
        required_size = num_elements * element_size

        if required_size > slot.size:
            raise ValueError(
                f"Requested tensor size {required_size} exceeds slot size {slot.size}"
            )

        buffer = (ctypes.c_byte * slot.size).from_address(addr)
        buffer_tensor = torch.frombuffer(
            buffer, dtype=dtype, count=num_elements
        ).reshape(shape)

        if not copy:
            return buffer_tensor

        if device is None:
            device = self.device

        target_tensor = torch.empty(shape, dtype=dtype, device=device)
        target_tensor.copy_(buffer_tensor)
        return target_tensor

    def cleanup(self) -> None:
        """Clean up all resources."""
        with self._lock:
            self._allocated.clear()
            self._offset_index.clear()
            self._pin_counts.clear()
            self._next_offset = 0
        logger.debug("EC Buffer cleaned up")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def used_size(self) -> int:
        """Currently used buffer size in bytes (live allocations only)."""
        with self._lock:
            return sum(slot.size for slot in self._allocated.values())

    @property
    def free_size(self) -> int:
        """Available buffer size in bytes."""
        return self.buffer_size - self.used_size

    @property
    def num_allocated(self) -> int:
        """Number of live allocated slots."""
        with self._lock:
            return len(self._allocated)

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _find_alloc_offset(self, size: int) -> int | None:
        """Find the first ring-order range that does not overlap pinned slots."""
        start = self._next_offset if self._next_offset < self.buffer_size else 0
        offset = self._find_alloc_offset_from(start, size)
        if offset is not None:
            return offset
        if start != 0:
            logger.debug(
                "Ring wrap: _next_offset=%d, size=%d, buffer_size=%d",
                self._next_offset,
                size,
                self.buffer_size,
            )
            return self._find_alloc_offset_from(0, size)
        return None

    def _find_alloc_offset_from(self, start: int, size: int) -> int | None:
        """Find a candidate offset at or after start.

        Unpinned slots do not block allocation because they can be evicted.
        Caller must hold self._lock.
        """
        offset = start
        if offset + size > self.buffer_size:
            return None

        pinned_ranges = []
        for addr, pin_count in self._pin_counts.items():
            if pin_count <= 0:
                continue
            slot = self._allocated.get(addr)
            if slot is not None:
                pinned_ranges.append((slot.offset, slot.offset + slot.size))
        pinned_ranges.sort()

        for pinned_start, pinned_end in pinned_ranges:
            if pinned_end <= offset:
                continue
            if pinned_start >= offset + size:
                return offset
            offset = max(offset, pinned_end)
            if offset + size > self.buffer_size:
                return None

        return offset

    def _evict_range(self, start: int, end: int) -> list[int]:
        """Evict all unpinned live slots overlapping [start, end).

        Caller must hold self._lock.
        """
        victims: list[tuple[int, int]] = []
        for offset, addr in self._offset_index:
            slot = self._allocated[addr]
            if self._ranges_overlap(start, end, offset, offset + slot.size):
                if self._pin_counts.get(addr, 0) > 0:
                    raise BufferError(
                        f"Cannot evict pinned EC transfer buffer slot 0x{addr:x}"
                    )
                victims.append((offset, addr))

        if not victims:
            return []

        victim_set = set(victims)
        self._offset_index = [
            item for item in self._offset_index if item not in victim_set
        ]

        evicted_addrs = []
        for offset, addr in victims:
            slot = self._allocated.pop(addr)
            self._pin_counts.pop(addr, None)
            evicted_addrs.append(addr)
            logger.debug(
                "Evicted slot (ring): addr=0x%x, offset=%d, size=%d",
                addr,
                offset,
                slot.size,
            )
        return evicted_addrs

    @staticmethod
    def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return start_a < end_b and start_b < end_a

    def _notify_free(self, addrs: list[int]) -> None:
        if self.on_free is None:
            return
        for addr in addrs:
            self.on_free(addr)

    def _remove_from_index(self, offset: int, addr: int) -> None:
        """Remove a single entry from the sorted offset index.

        Caller must hold self._lock.
        """
        lo = bisect.bisect_left(self._offset_index, (offset, addr))
        if lo < len(self._offset_index) and self._offset_index[lo] == (offset, addr):
            del self._offset_index[lo]
