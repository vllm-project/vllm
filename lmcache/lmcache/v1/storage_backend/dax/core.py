# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Optional, TypeVar
import ctypes
import enum
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)

KeyT = TypeVar("KeyT", bound=Hashable)
_CopyFn = Callable[[int, MemoryObj, int], None]


class DaxPutFromPtrResult(enum.Enum):
    """Result of copying a reserved source payload into a DAX slot."""

    INSERTED = enum.auto()
    ALREADY_EXISTS = enum.auto()
    INFLIGHT = enum.auto()
    NO_SPACE = enum.auto()
    CLOSED = enum.auto()
    INVALID = enum.auto()
    COPY_FAILED = enum.auto()


@dataclass
class _Entry:
    offset: int
    meta: DiskCacheMetadata
    slot_id: int
    generation: int


@dataclass
class _Inflight:
    offset: int
    meta: DiskCacheMetadata
    slot_id: int
    generation: int
    canceled: bool = False


@dataclass
class _SlotState:
    generation: int
    committed: bool = False
    borrow_count: int = 0
    pending_free: bool = False


@dataclass(frozen=True)
class DaxReadReservation(Generic[KeyT]):
    """A committed DAX entry reserved for a read or restore operation."""

    result_index: int
    key: KeyT
    offset: int
    size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor]
    slot_id: int
    generation: int


class DaxCore(Generic[KeyT]):
    """Thread-safe shared core for DAX-backed fixed-slot storage.

    ``DaxCore`` owns the mapped arena, in-memory key index, slot allocator,
    external lock refcounts, and read borrow counts. Eviction policy is owned
    by LMCache controllers, not by the DAX core.
    """

    def __init__(
        self,
        *,
        device_path: str,
        max_dax_size_bytes: int,
        slot_bytes: int,
    ) -> None:
        """Open and map a DAX arena.

        Args:
            device_path: Path to the mmap-able DAX device or file.
            max_dax_size_bytes: Number of bytes to map from the device.
            slot_bytes: Fixed slot size used for each stored object.

        Raises:
            ValueError: If the path is empty, sizes are non-positive, or the
                arena cannot fit at least one slot.
            RuntimeError: If the device cannot be opened or mapped, or if
                ``max_dax_size_bytes`` exceeds the reported device capacity.
        """
        self._device_path = device_path
        self._arena_bytes = max_dax_size_bytes
        self._slot_bytes = slot_bytes

        if not self._device_path:
            raise ValueError("device_path must be a non-empty string")
        if self._arena_bytes <= 0:
            raise ValueError("max_dax_size_bytes must be > 0")
        if self._slot_bytes <= 0:
            raise ValueError("slot_bytes must be > 0")

        self._fd: Optional[int] = None
        self._mmap_obj: Optional[mmap.mmap] = None
        self._arena_view: Optional[memoryview] = None
        self._base_ptr: int = 0

        self._state_lock = threading.RLock()
        self._state_condition = threading.Condition(self._state_lock)

        self._index: dict[KeyT, _Entry] = {}
        self._external_lock_counts: dict[KeyT, int] = {}
        self._inflight: dict[KeyT, _Inflight] = {}
        self._slot_states: dict[int, _SlotState] = {}

        self._next_slot = 0
        self._free_slots: set[int] = set()
        self._active_writes = 0
        self._active_reads = 0
        self._closing = False
        self._closed = False

        self._open_arena()
        self._max_slots = self._arena_bytes // self._slot_bytes
        if self._max_slots <= 0:
            self.close()
            raise ValueError("configured arena does not fit even one slot")

    @property
    def device_path(self) -> str:
        """Return the path used to open the DAX arena."""
        return self._device_path

    @property
    def arena_bytes(self) -> int:
        """Return the mapped arena size in bytes."""
        return self._arena_bytes

    @property
    def slot_bytes(self) -> int:
        """Return the fixed slot size in bytes."""
        return self._slot_bytes

    @property
    def max_slots(self) -> int:
        """Return the maximum number of slots in the arena."""
        return self._max_slots

    @property
    def fd(self) -> Optional[int]:
        """Return the open device file descriptor, or ``None`` after close."""
        return self._fd

    @property
    def mmap_obj(self) -> Optional[mmap.mmap]:
        """Return the active mmap object, or ``None`` after close."""
        return self._mmap_obj

    @property
    def arena_view(self) -> Optional[memoryview]:
        """Return the active arena memoryview, or ``None`` after close."""
        return self._arena_view

    @property
    def base_ptr(self) -> int:
        """Return the base virtual address of the mapped arena."""
        return self._base_ptr

    def is_closing(self) -> bool:
        """Check whether the core is closing or already closed.

        Returns:
            ``True`` after close has begun, otherwise ``False``.
        """
        with self._state_lock:
            return self._closing or self._closed

    def has_inflight(self, key: KeyT) -> bool:
        """Check whether a key currently has an uncommitted write.

        Args:
            key: Key to check.

        Returns:
            ``True`` if a write reservation exists for ``key``.
        """
        with self._state_lock:
            return key in self._inflight

    def put_many(
        self,
        keys: list[KeyT],
        objs: list[MemoryObj],
    ) -> list[bool]:
        """Store multiple memory objects in DAX slots.

        Existing committed or in-flight keys are treated as no-op successes.
        Unsupported multi-tensor objects, oversized objects, copy failures,
        and closing state produce ``False`` for the affected key.

        Args:
            keys: Keys to store.
            objs: Memory objects whose contents are copied into DAX.

        Returns:
            Per-key success flags aligned with ``keys``.

        Raises:
            ValueError: If ``keys`` and ``objs`` have different lengths.
        """
        if len(keys) != len(objs):
            raise ValueError(
                "keys and objs must have the same length, "
                f"got {len(keys)} and {len(objs)}"
            )

        results: list[bool] = []
        for key, obj in zip(keys, objs, strict=True):
            results.append(
                self._put_one(
                    key,
                    obj,
                    raise_on_full=False,
                    raise_on_write_failure=False,
                )
            )
        return results

    def put_one(
        self,
        key: KeyT,
        obj: MemoryObj,
        *,
        writer: Optional[_CopyFn] = None,
    ) -> bool:
        """Store one memory object in a DAX slot.

        This method exists for wrappers that need the existing non-MP
        single-object error semantics. Slot exhaustion and copy failures raise
        ``RuntimeError``. MP code should generally use ``put_many`` to
        preserve per-key success reporting.

        Args:
            key: Key to store.
            obj: Memory object whose contents are copied into DAX.
            writer: Optional copy function accepting
                ``(dax_offset, memory_obj, size)``. If omitted, direct
                ``ctypes.memmove`` is used.

        Returns:
            ``True`` if the key was committed or was already committed or
            in-flight; ``False`` if the object could not be stored.

        Raises:
            RuntimeError: If no slot can be allocated or the copy fails.
        """
        return self._put_one(
            key,
            obj,
            raise_on_full=True,
            raise_on_write_failure=True,
            writer=writer,
        )

    def exists_many(
        self,
        keys: list[KeyT],
        lock: bool = False,
    ) -> list[bool]:
        """Check whether keys exist, optionally acquiring external locks.

        Args:
            keys: Keys to look up.
            lock: If ``True``, increment the external lock refcount for each
                found key.

        Returns:
            Per-key hit flags aligned with ``keys``.
        """
        results = [False] * len(keys)
        with self._state_lock:
            if self._closing or self._closed:
                return results

            for i, key in enumerate(keys):
                entry = self._index.get(key)
                if entry is None:
                    continue
                if self._get_committed_state_locked(entry) is None:
                    continue
                results[i] = True
                if lock:
                    self._external_lock_counts[key] = (
                        self._external_lock_counts.get(key, 0) + 1
                    )
        return results

    def load_many_into(
        self,
        keys: list[KeyT],
        objs: list[MemoryObj],
    ) -> list[bool]:
        """Load DAX entries directly into caller-provided buffers.

        Args:
            keys: Keys to load.
            objs: Destination memory objects. Each destination must have
                enough capacity for the corresponding stored payload.

        Returns:
            Per-key success flags aligned with ``keys``.

        Raises:
            ValueError: If ``keys`` and ``objs`` have different lengths.
        """
        if len(keys) != len(objs):
            raise ValueError(
                "keys and objs must have the same length, "
                f"got {len(keys)} and {len(objs)}"
            )

        reservations, _ = self.reserve_reads(keys, prefix_only=False)
        results = [False] * len(keys)
        touched_keys: set[KeyT] = set()

        try:
            for reservation in reservations:
                obj = objs[reservation.result_index]
                if len(obj.get_shapes()) > 1:
                    continue
                if int(obj.get_size()) < reservation.size:
                    continue

                try:
                    self._default_do_read(reservation.offset, obj, reservation.size)
                except Exception:
                    logger.exception(
                        "Failed to load DAX key %s into caller buffer",
                        reservation.key,
                    )
                    continue

                obj.metadata.cached_positions = reservation.cached_positions
                results[reservation.result_index] = True
                touched_keys.add(reservation.key)
        finally:
            self.finalize_reads(reservations, touched_keys)

        return results

    def snapshot_keys(self) -> list[KeyT]:
        """Return committed keys currently indexed by this DAX arena.

        Returns:
            A snapshot list of keys whose slots are committed and readable at
            the time of the call. Later writes, deletes, or migration steps do
            not mutate the returned list.
        """
        with self._state_lock:
            if self._closing or self._closed:
                return []

            return [
                key
                for key, entry in self._index.items()
                if self._get_committed_state_locked(entry) is not None
            ]

    def reserve_reads_for_keys(
        self,
        keys: list[KeyT],
    ) -> list[DaxReadReservation[KeyT]]:
        """Reserve readable slots for explicit keys.

        Args:
            keys: Keys to reserve for a lock-safe read or migration copy.

        Returns:
            Reservations for keys that were present and readable. The caller
            must pass the returned reservations to ``finalize_reads``.
        """
        reservations, _ = self.reserve_reads(keys, prefix_only=False)
        return reservations

    def put_reserved_from_ptr(
        self,
        key: KeyT,
        src_ptr: int,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: torch.Tensor | None,
    ) -> DaxPutFromPtrResult:
        """Store one key by copying from an already reserved source pointer.

        Args:
            key: Destination key to commit.
            src_ptr: Source memory address. The caller is responsible for
                keeping this pointer valid until the call returns.
            size: Number of bytes to copy.
            shape: Tensor shape to record for future reads.
            dtype: Tensor dtype to record for future reads.
            fmt: Memory format to record for future reads.
            cached_positions: Optional cached-position metadata.

        Returns:
            A ``DaxPutFromPtrResult`` describing whether the payload was newly
            inserted, already present, full, closed, invalid, or failed.
        """
        if size <= 0 or size > self._slot_bytes:
            return DaxPutFromPtrResult.INVALID
        if src_ptr <= 0:
            return DaxPutFromPtrResult.INVALID

        with self._state_lock:
            if self._closing or self._closed:
                return DaxPutFromPtrResult.CLOSED
            if key in self._index:
                return DaxPutFromPtrResult.ALREADY_EXISTS
            if key in self._inflight:
                return DaxPutFromPtrResult.INFLIGHT

            try:
                slot_id = self._allocate_slot_locked()
            except RuntimeError:
                return DaxPutFromPtrResult.NO_SPACE

            offset = slot_id * self._slot_bytes
            generation = self._reserve_slot_state_locked(slot_id)
            meta = DiskCacheMetadata(
                path=f"{self._device_path}@{offset}",
                size=size,
                shape=shape,
                dtype=dtype,
                cached_positions=cached_positions,
                fmt=fmt,
                pin_count=0,
            )
            self._inflight[key] = _Inflight(
                offset=offset,
                meta=meta,
                slot_id=slot_id,
                generation=generation,
            )
            self._active_writes += 1
            dst_ptr = self._base_ptr + offset

        write_failed = False
        try:
            ctypes.memmove(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                size,
            )
        except Exception:
            logger.exception("Failed to copy reserved DAX payload for key %s", key)
            write_failed = True

        if self._finalize_put(key, write_failed=write_failed):
            return DaxPutFromPtrResult.INSERTED
        return DaxPutFromPtrResult.COPY_FAILED

    def unlock_many(self, keys: list[KeyT]) -> None:
        """Release external locks for keys.

        Args:
            keys: Keys whose external lock refcount should be decremented.
        """
        with self._state_lock:
            for key in keys:
                count = self._external_lock_counts.get(key, 0)
                if count <= 1:
                    self._external_lock_counts.pop(key, None)
                else:
                    self._external_lock_counts[key] = count - 1

    def delete_many(
        self,
        keys: list[KeyT],
        *,
        force: bool = False,
    ) -> list[bool]:
        """Delete keys from the in-memory DAX index.

        When ``force`` is ``False``, externally locked and in-flight keys are
        skipped. Slots borrowed by active reads are marked for later reclaim
        and become reusable after ``finalize_reads`` drains the borrow count.

        Args:
            keys: Keys to delete.
            force: If ``True``, remove externally locked keys and cancel
                in-flight writes. This is used by the non-MP wrapper to keep
                its existing force-remove behavior.

        Returns:
            Per-key deletion flags aligned with ``keys``.
        """
        results = [False] * len(keys)
        with self._state_lock:
            if self._closed:
                return results

            for i, key in enumerate(keys):
                inflight = self._inflight.get(key)
                if inflight is not None:
                    if not force:
                        continue
                    inflight.canceled = True
                    self._external_lock_counts.pop(key, None)
                    results[i] = True
                    continue

                entry = self._index.get(key)
                if entry is None:
                    continue

                if not force and self._external_lock_counts.get(key, 0) > 0:
                    continue

                self._index.pop(key, None)
                self._external_lock_counts.pop(key, None)
                self._schedule_slot_reclaim_locked(entry.slot_id, entry.generation)
                results[i] = True

        return results

    def usage(self) -> tuple[float, float]:
        """Return current and post-eviction slot usage fractions.

        Returns:
            ``(current_usage, usage_after_ongoing_eviction)`` where both values
            are fractions of total slots. The second value subtracts slots
            already removed from the index but still waiting for read borrows
            to drain.
        """
        with self._state_lock:
            if self._max_slots <= 0:
                return (0.0, 0.0)

            live_slot_count = self._next_slot - len(self._free_slots)
            pending_reclaim_count = sum(
                1 for state in self._slot_states.values() if state.pending_free
            )
            current_usage = live_slot_count / self._max_slots
            post_eviction_slots = max(0, live_slot_count - pending_reclaim_count)
            usage_after_eviction = post_eviction_slots / self._max_slots
            return (current_usage, usage_after_eviction)

    def reserve_reads(
        self,
        keys: list[KeyT],
        *,
        prefix_only: bool,
    ) -> tuple[list[DaxReadReservation[KeyT]], list[bool]]:
        """Reserve committed entries for a read pipeline.

        Each reservation increments the slot borrow count so eviction can
        remove the key from the index without recycling the slot until
        ``finalize_reads`` is called.

        Args:
            keys: Keys to reserve.
            prefix_only: If ``True``, stop at the first missing or unreadable
                key. If ``False``, continue and leave holes in the result
                bitmap.

        Returns:
            A tuple of read reservations and per-key hit flags aligned with
            ``keys``.
        """
        reservations: list[DaxReadReservation[KeyT]] = []
        results = [False] * len(keys)

        with self._state_lock:
            if self._closing or self._closed:
                return reservations, results

            for i, key in enumerate(keys):
                entry = self._index.get(key)
                if entry is None:
                    if prefix_only:
                        break
                    continue

                meta = entry.meta
                if meta.shape is None or meta.dtype is None or meta.fmt is None:
                    if prefix_only:
                        break
                    continue

                state = self._get_committed_state_locked(entry)
                if state is None:
                    if prefix_only:
                        break
                    continue

                state.borrow_count += 1
                reservations.append(
                    DaxReadReservation(
                        result_index=i,
                        key=key,
                        offset=entry.offset,
                        size=int(meta.size),
                        shape=meta.shape,
                        dtype=meta.dtype,
                        fmt=meta.fmt,
                        cached_positions=meta.cached_positions,
                        slot_id=entry.slot_id,
                        generation=entry.generation,
                    )
                )
                results[i] = True

            if reservations:
                self._active_reads += 1

        return reservations, results

    def finalize_reads(
        self,
        reservations: list[DaxReadReservation[KeyT]],
        touched_keys: set[KeyT],
    ) -> None:
        """Release read reservations.

        Args:
            reservations: Reservations previously returned by
                ``reserve_reads``.
            touched_keys: Subset of reservation keys whose data was
                successfully copied. Kept for callers that already track
                successful reads; DAX core does not own eviction recency.
        """
        del touched_keys
        if not reservations:
            return

        with self._state_lock:
            if self._active_reads > 0:
                self._active_reads -= 1
            else:
                logger.warning("DaxCore active read count underflow")

            for reservation in reservations:
                state = self._slot_states.get(reservation.slot_id)
                if state is None or state.generation != reservation.generation:
                    continue

                if state.borrow_count > 0:
                    state.borrow_count -= 1

                if state.pending_free and state.borrow_count == 0:
                    state.pending_free = False
                    self._free_slot_locked(reservation.slot_id)

            self._state_condition.notify_all()

    def can_shrink_to(self, new_max_slots: int) -> bool:
        """Check whether every live or borrowed slot fits below a new limit.

        Args:
            new_max_slots: Candidate slot count after a shrink.

        Returns:
            ``True`` if remapping to ``new_max_slots`` would not invalidate
            committed, in-flight, or borrowed slot state.
        """
        if new_max_slots <= 0:
            return False

        with self._state_lock:
            if self._closed:
                return False

            for entry in self._index.values():
                if (
                    self._get_committed_state_locked(entry) is not None
                    and entry.slot_id >= new_max_slots
                ):
                    return False
            for inflight in self._inflight.values():
                if inflight.slot_id >= new_max_slots and not inflight.canceled:
                    return False
            for slot_id, state in self._slot_states.items():
                if slot_id >= new_max_slots and state.borrow_count > 0:
                    return False
            return True

    def remap(self, new_max_dax_size_bytes: int) -> None:
        """Remap this arena to a new byte size while preserving metadata.

        Args:
            new_max_dax_size_bytes: New number of bytes to map.

        Raises:
            ValueError: If the new size cannot fit at least one slot or would
                cut off live slot metadata.
            RuntimeError: If the arena is closed or the mmap operation fails.
        """
        if new_max_dax_size_bytes <= 0:
            raise ValueError("new_max_dax_size_bytes must be > 0")
        new_max_slots = new_max_dax_size_bytes // self._slot_bytes
        if new_max_slots <= 0:
            raise ValueError("new DAX arena does not fit even one slot")

        old_fd: Optional[int] = None
        old_mmap_obj: Optional[mmap.mmap] = None
        old_arena_view: Optional[memoryview] = None

        with self._state_lock:
            if self._closed:
                raise RuntimeError("DaxCore is closed")

            while self._active_writes > 0 or self._active_reads > 0:
                if not self._state_condition.wait(timeout=30.0):
                    logger.warning(
                        "DaxCore remap: still waiting for %d writes, %d reads",
                        self._active_writes,
                        self._active_reads,
                    )

            if not self.can_shrink_to(new_max_slots):
                raise ValueError("new DAX arena would cut off live slots")

            old_fd = self._fd
            old_mmap_obj = self._mmap_obj
            old_arena_view = self._arena_view
            old_arena_bytes = self._arena_bytes
            old_max_slots = self._max_slots
            old_base_ptr = self._base_ptr

            self._fd = None
            self._mmap_obj = None
            self._arena_view = None
            self._base_ptr = 0
            self._arena_bytes = new_max_dax_size_bytes
            self._max_slots = new_max_slots

            try:
                self._open_arena()
            except Exception:
                self._fd = old_fd
                self._mmap_obj = old_mmap_obj
                self._arena_view = old_arena_view
                self._arena_bytes = old_arena_bytes
                self._max_slots = old_max_slots
                self._base_ptr = old_base_ptr
                raise

            self._free_slots = {
                slot_id for slot_id in self._free_slots if slot_id < new_max_slots
            }
            self._next_slot = min(self._next_slot, new_max_slots)
            self._state_condition.notify_all()

        self._release_arena_resources(old_fd, old_mmap_obj, old_arena_view)

    def close(self) -> None:
        """Close the DAX arena after active reads and writes finish.

        The call is idempotent. It marks the core as closing, waits for active
        operations to drain, clears volatile in-memory indexes, and releases
        the mmap, memoryview, and file descriptor.
        """
        fd = None
        mmap_obj = None
        arena_view = None

        with self._state_lock:
            if self._closed:
                return

            self._closing = True
            while self._active_writes > 0 or self._active_reads > 0:
                if not self._state_condition.wait(timeout=30.0):
                    logger.warning(
                        "DaxCore close: still waiting for %d writes, %d reads",
                        self._active_writes,
                        self._active_reads,
                    )

            if self._closed:
                return

            self._closed = True
            self._index.clear()
            self._external_lock_counts.clear()
            self._inflight.clear()
            self._slot_states.clear()
            self._free_slots.clear()
            self._next_slot = 0

            fd = self._fd
            mmap_obj = self._mmap_obj
            arena_view = self._arena_view
            self._fd = None
            self._mmap_obj = None
            self._arena_view = None
            self._base_ptr = 0

        self._release_arena_resources(fd, mmap_obj, arena_view)

    def report_status(self) -> dict:
        """Return a status snapshot for health and observability.

        Returns:
            Dictionary containing health, capacity, slot occupancy, external
            lock count, borrowed slot count, close state, and restart-recovery
            support.
        """
        with self._state_lock:
            live_slot_count = self._next_slot - len(self._free_slots)
            borrowed_slot_count = sum(
                1 for state in self._slot_states.values() if state.borrow_count > 0
            )
            return {
                "is_healthy": not self._closed and self._mmap_obj is not None,
                "device_path": self._device_path,
                "max_dax_size_bytes": self._arena_bytes,
                "slot_bytes": self._slot_bytes,
                "max_slots": self._max_slots,
                "live_slot_count": live_slot_count,
                "locked_key_count": len(self._external_lock_counts),
                "borrowed_slot_count": borrowed_slot_count,
                "active_read_count": self._active_reads,
                "active_write_count": self._active_writes,
                "closing": self._closing or self._closed,
                "supports_restart_recovery": False,
            }

    def _put_one(
        self,
        key: KeyT,
        obj: MemoryObj,
        *,
        raise_on_full: bool,
        raise_on_write_failure: bool,
        writer: Optional[_CopyFn] = None,
    ) -> bool:
        if len(obj.get_shapes()) > 1:
            logger.error(
                "DaxCore does not support multi-tensor allocations for key %s",
                key,
            )
            return False

        size = int(obj.get_size())
        if size > self._slot_bytes:
            return False

        reserve_result = self._reserve_put(
            key,
            obj,
            size,
            raise_on_full=raise_on_full,
        )
        if reserve_result is None:
            return False
        if reserve_result is True:
            return True

        assert isinstance(reserve_result, _Inflight)
        copy_fn = writer or self._default_do_write
        write_failed = False
        write_error: Optional[Exception] = None
        try:
            copy_fn(reserve_result.offset, obj, size)
        except Exception as exc:
            write_failed = True
            write_error = exc

        committed = self._finalize_put(key, write_failed=write_failed)
        if write_error is not None and raise_on_write_failure:
            raise RuntimeError(
                f"DAX write failed for key {key}: {write_error}"
            ) from write_error
        return committed

    def _reserve_put(
        self,
        key: KeyT,
        obj: MemoryObj,
        size: int,
        *,
        raise_on_full: bool,
    ) -> _Inflight | bool | None:
        with self._state_lock:
            if self._closing or self._closed:
                return None

            if key in self._index or key in self._inflight:
                return True

            try:
                slot_id = self._allocate_slot_locked()
            except RuntimeError:
                if raise_on_full:
                    raise
                return None

            offset = slot_id * self._slot_bytes
            generation = self._reserve_slot_state_locked(slot_id)
            meta = DiskCacheMetadata(
                path=f"{self._device_path}@{offset}",
                size=size,
                shape=obj.metadata.shape,
                dtype=obj.metadata.dtype,
                cached_positions=obj.metadata.cached_positions,
                fmt=obj.metadata.fmt,
                pin_count=0,
            )
            inflight = _Inflight(
                offset=offset,
                meta=meta,
                slot_id=slot_id,
                generation=generation,
            )
            self._inflight[key] = inflight
            self._active_writes += 1
            return inflight

    def _finalize_put(self, key: KeyT, *, write_failed: bool) -> bool:
        with self._state_lock:
            committed = self._finalize_inflight_locked(key, write_failed=write_failed)
            if self._active_writes > 0:
                self._active_writes -= 1
            else:
                logger.warning("DaxCore active write count underflow for key %s", key)
            self._state_condition.notify_all()
            return committed

    def _reserve_slot_state_locked(self, slot_id: int) -> int:
        existing = self._slot_states.get(slot_id)
        new_generation = (existing.generation if existing is not None else 0) + 1
        self._slot_states[slot_id] = _SlotState(generation=new_generation)
        return new_generation

    def _mark_slot_committed_locked(self, slot_id: int, generation: int) -> None:
        state = self._slot_states.get(slot_id)
        if state is None or state.generation != generation:
            return
        state.committed = True
        state.pending_free = False

    def _schedule_slot_reclaim_locked(self, slot_id: int, generation: int) -> None:
        state = self._slot_states.get(slot_id)
        if state is None or state.generation != generation:
            return
        state.committed = False
        if state.borrow_count == 0:
            state.pending_free = False
            self._free_slot_locked(slot_id)
        else:
            state.pending_free = True

    def _get_committed_state_locked(self, entry: _Entry) -> Optional[_SlotState]:
        state = self._slot_states.get(entry.slot_id)
        if state is None or state.generation != entry.generation or not state.committed:
            return None
        return state

    def _finalize_inflight_locked(self, key: KeyT, *, write_failed: bool) -> bool:
        inflight = self._inflight.pop(key, None)
        if inflight is None:
            return False

        if inflight.canceled or write_failed:
            self._schedule_slot_reclaim_locked(inflight.slot_id, inflight.generation)
            return False

        self._mark_slot_committed_locked(inflight.slot_id, inflight.generation)
        self._index[key] = _Entry(
            offset=inflight.offset,
            meta=inflight.meta,
            slot_id=inflight.slot_id,
            generation=inflight.generation,
        )
        return True

    def _allocate_slot_locked(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot < self._max_slots:
            slot_id = self._next_slot
            self._next_slot += 1
            return slot_id
        raise RuntimeError("No free slots available; eviction required")

    def _free_slot_locked(self, slot_id: int) -> None:
        if slot_id >= 0:
            self._free_slots.add(slot_id)

    def _default_do_write(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(self._base_ptr + offset),
            ctypes.c_void_p(memory_obj.data_ptr),
            size,
        )

    def _default_do_read(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(memory_obj.data_ptr),
            ctypes.c_void_p(self._base_ptr + offset),
            size,
        )

    def _open_arena(self) -> None:
        fd: Optional[int] = None
        mmap_obj: Optional[mmap.mmap] = None
        arena_view: Optional[memoryview] = None

        try:
            fd = os.open(self._device_path, os.O_RDWR)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open dax device {self._device_path}: {exc}"
            ) from exc

        try:
            try:
                capacity_bytes = os.fstat(fd).st_size
                if capacity_bytes > 0 and self._arena_bytes > capacity_bytes:
                    raise RuntimeError(
                        f"max_dax_size_bytes ({self._arena_bytes} bytes) exceeds "
                        f"device capacity ({capacity_bytes} bytes)"
                    )
            except OSError:
                logger.warning(
                    "Could not determine DAX device capacity via fstat; "
                    "skipping max_dax_size_bytes validation"
                )

            mmap_obj = mmap.mmap(
                fd,
                self._arena_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mmap_obj))
            arena_view = memoryview(mmap_obj)
            self._fd = fd
            self._mmap_obj = mmap_obj
            self._arena_view = arena_view
            self._base_ptr = base_ptr
        except Exception as exc:
            self._release_arena_resources(fd, mmap_obj, arena_view)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(
                f"Failed to mmap dax arena ({self._arena_bytes} bytes) from "
                f"{self._device_path}: {exc}"
            ) from exc

    @staticmethod
    def _release_arena_resources(
        fd: Optional[int],
        mmap_obj: Optional[mmap.mmap],
        arena_view: Optional[memoryview],
    ) -> None:
        if arena_view is not None:
            try:
                arena_view.release()
            except Exception as exc:
                logger.warning("Failed to release DAX memoryview: %s", exc)

        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception as exc:
                logger.warning("Failed to close DAX mmap: %s", exc)

        if fd is not None:
            try:
                os.close(fd)
            except Exception as exc:
                logger.warning("Failed to close DAX fd: %s", exc)
