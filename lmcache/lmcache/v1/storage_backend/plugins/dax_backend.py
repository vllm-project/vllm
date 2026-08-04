# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, cast
import asyncio
import ctypes
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from lmcache.v1.storage_backend.dax.core import DaxCore
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # Standard
    from concurrent.futures import Future

    # Third Party
    import torch

    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


@dataclass
class _RestoreItem:
    """Reserved DAX read metadata for one item in a batched restore."""

    result_index: int
    key: CacheEngineKey
    offset: int
    size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor]
    slot_id: int
    generation: int
    memory_obj: Optional[MemoryObj] = None
    slab_offset: int = 0


@dataclass
class _RestoreSpan:
    """One contiguous source span copied from DAX into the staging slab."""

    src_offset: int
    slab_offset: int
    size: int


@dataclass
class _RestoreRegion:
    """One restore region executed by a persistent worker."""

    region_index: int
    slab_offset: int
    total_bytes: int
    items: list[_RestoreItem]
    spans: list[_RestoreSpan]


@dataclass
class _RestoreWave:
    """One wave of region work against the fixed-size retrieve slab."""

    regions: list[_RestoreRegion]


class DaxBackend(StoragePluginInterface):
    """Storage plugin backend for /dev/dax mmap-backed KV cache."""

    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend: Optional[LocalCPUBackend] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ) -> None:
        """Initialize a non-MP DAX storage backend.

        Args:
            config: LMCache engine config containing ``dax.*`` extra config.
            metadata: Runtime metadata used to validate TP and KV layout.
            local_cpu_backend: CPU allocator used for restore outputs.
            loop: Event loop used when ``dax.async_put`` is enabled.
            dst_device: Destination device name passed to the base backend.

        Raises:
            ValueError: If required config, metadata, local CPU backend, or
                DAX options are missing or unsupported.
            RuntimeError: If the DAX arena cannot be opened or mapped.
        """
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.config is None:
            raise ValueError("DaxBackend requires config")
        if self.metadata is None:
            raise ValueError("DaxBackend requires metadata")

        if self.metadata.world_size != 1:
            raise ValueError(
                "DaxBackend currently only supports TP=1 "
                f"(world_size={self.metadata.world_size})"
            )
        if self.metadata.get_num_groups() != 1:
            raise ValueError(
                "DaxBackend currently supports only single-group KV layout"
            )

        extra = self.config.extra_config or {}
        self.device_path = str(extra.get("dax.device_path", "")).strip()
        if not self.device_path:
            raise ValueError("extra_config['dax.device_path'] is required")

        self.async_put = _to_bool(extra.get("dax.async_put", False))
        if self.async_put and self.loop is None:
            raise ValueError("DaxBackend async_put=true requires an asyncio event loop")

        self.max_dax_size = float(extra.get("dax.max_dax_size", 0))
        if self.max_dax_size <= 0:
            raise ValueError("extra_config['dax.max_dax_size'] must be > 0")

        if self.local_cpu_backend is None:
            raise ValueError("DaxBackend requires local_cpu_backend")

        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False

        self._restore_executor: Optional[ThreadPoolExecutor] = None
        self._restore_dispatch_executor: Optional[ThreadPoolExecutor] = None
        self._retrieve_staging_slab_ptr: int = 0
        self._retrieve_staging_slab_bytes: int = 0
        self._restore_region_bytes: int = 0
        self._restore_workers: int = 0
        self._restore_max_regions: int = 0

        full_chunk_size = int(self.local_cpu_backend.get_full_chunk_size_bytes())
        self._core = DaxCore[CacheEngineKey](
            device_path=self.device_path,
            max_dax_size_bytes=int(self.max_dax_size * 1024**3),
            slot_bytes=max(1, full_chunk_size),
        )
        self.slot_bytes = self._core.slot_bytes
        self._arena_bytes = self._core.arena_bytes
        self._max_slots = self._core.max_slots

        try:
            default_restore_workers = min(8, max(1, os.cpu_count() or 1))
            self._restore_workers = self._get_positive_int_extra(
                extra,
                "dax.restore_workers",
                default_restore_workers,
            )
            self._restore_max_regions = self._get_positive_int_extra(
                extra,
                "dax.restore_max_regions",
                self._restore_workers,
            )
            default_staging_slab_bytes = max(
                256 * 1024 * 1024,
                self._restore_max_regions * self.slot_bytes,
            )
            self._retrieve_staging_slab_bytes = self._get_positive_int_extra(
                extra,
                "dax.retrieve_staging_slab_bytes",
                default_staging_slab_bytes,
            )
            min_required_slab = self._restore_max_regions * self.slot_bytes
            if self._retrieve_staging_slab_bytes < min_required_slab:
                raise ValueError(
                    "extra_config['dax.retrieve_staging_slab_bytes'] must be at "
                    f"least {min_required_slab} bytes"
                )
            self._restore_region_bytes = (
                self._retrieve_staging_slab_bytes // self._restore_max_regions
            )
            if self._restore_region_bytes < self.slot_bytes:
                raise ValueError(
                    "dax.retrieve_staging_slab_bytes does not leave enough space "
                    "per restore region for one full chunk"
                )

            self._retrieve_staging_slab_ptr = int(
                lmc_ops.alloc_pinned_ptr(self._retrieve_staging_slab_bytes, 0)
            )
            self._restore_executor = ThreadPoolExecutor(
                max_workers=self._restore_workers,
                thread_name_prefix="dax-restore",
            )
            self._restore_dispatch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="dax-restore-dispatch",
            )
        except Exception:
            self._release_restore_resources()
            self._core.close()
            raise

        logger.info(
            "DaxBackend init: device=%s dax_size=%d slot=%d max_slots=%d "
            "restore_workers=%d restore_regions=%d restore_slab=%d",
            self.device_path,
            self._arena_bytes,
            self.slot_bytes,
            self._max_slots,
            self._restore_workers,
            self._restore_max_regions,
            self._retrieve_staging_slab_bytes,
        )

    def __str__(self) -> str:
        return "DaxBackend"

    @property
    def _fd(self) -> Optional[int]:
        return self._core.fd

    @property
    def _mmap_obj(self):
        return self._core.mmap_obj

    @property
    def _base_ptr(self) -> int:
        return self._core.base_ptr

    @property
    def _arena_view(self) -> Optional[memoryview]:
        return self._core.arena_view

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether ``key`` exists in the backend.

        Args:
            key: The cache key to look up.
            pin: If ``True`` and the key is present, atomically
                increment its external lock count.

        Returns:
            ``True`` if the key is present, ``False`` otherwise.
        """
        return self._core.exists_many([key], lock=pin)[0]

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether ``key`` is tracked as an in-flight put task.

        Args:
            key: The cache key to check.

        Returns:
            ``True`` if ``key`` is in the in-flight put task set.
        """
        return self._core.has_inflight(key)

    def pin(self, key: CacheEngineKey) -> bool:
        """Increment the external lock count for ``key`` if it exists.

        Args:
            key: The cache key to pin.

        Returns:
            ``True`` if the key was found and pinned, ``False`` otherwise.
        """
        return self._core.exists_many([key], lock=True)[0]

    def unpin(self, key: CacheEngineKey) -> bool:
        """Decrement the external lock count for ``key``.

        Args:
            key: The cache key to unpin.

        Returns:
            ``True`` if ``key`` is present in the backend after the
            operation, ``False`` otherwise.
        """
        self._core.unlock_many([key])
        return self._core.exists_many([key], lock=False)[0]

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove ``key`` from the backend if present.

        If the key is in-flight, it is marked canceled. If it is
        committed, its slot is scheduled for reclamation according to
        the current slot state.

        Args:
            key: The cache key to remove.
            force: If ``False``, pinned keys are left untouched. If
                ``True``, the key is removed even when externally locked.

        Returns:
            ``True`` if the key was present or in-flight and removal was
            scheduled, ``False`` otherwise.
        """
        return self._core.delete_many([key], force=force)[0]

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """Store a batch of memory objects in the DAX arena.

        Args:
            keys: Cache keys to store.
            objs: Memory objects whose contents are copied into DAX slots.
            transfer_spec: Transfer hint accepted for interface compatibility
                and ignored by this backend.
            on_complete_callback: Optional callback invoked for each key that
                is newly committed.

        Returns:
            Async put futures when ``dax.async_put`` schedules work on the
            configured event loop; otherwise ``None``.

        Raises:
            ValueError: If input lengths differ or an object exceeds the slot
                size.
            RuntimeError: If the backend is closing or a synchronous DAX write
                cannot be committed.
        """
        del transfer_spec
        if len(keys) != len(objs):
            raise ValueError(
                "keys and objs must have the same length, "
                f"got {len(keys)} and {len(objs)}"
            )
        if self._is_closed_or_closing():
            raise RuntimeError("DaxBackend is closing")

        futures: list[Future] = []
        for key, obj in zip(keys, objs, strict=True):
            num_shapes = len(obj.get_shapes())
            if num_shapes > 1:
                logger.error(
                    "DaxBackend does not support multi-tensor allocations: "
                    "key=%s has %d tensors. "
                    "Use single-tensor format or extend metadata.",
                    key,
                    num_shapes,
                )
                continue

            size = int(obj.get_size())
            if size > self.slot_bytes:
                raise ValueError(
                    f"DaxBackend: object size {size} for key {key} "
                    f"exceeds slot size {self.slot_bytes}"
                )

            # Preserve current overwrite posture: duplicates are a no-op success
            # without firing completion callbacks.
            if self.contains(key) or self.exists_in_put_tasks(key):
                continue

            if self.async_put and self.loop is not None and self.loop.is_running():
                obj.ref_count_up()
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._submit_write(
                            key=key,
                            memory_obj=obj,
                            on_complete_callback=on_complete_callback,
                        ),
                        self.loop,
                    )
                except Exception:
                    obj.ref_count_down()
                    raise
                futures.append(future)
                continue

            committed = self._core.put_one(
                key,
                obj,
                writer=lambda offset, memory_obj, copy_size: self._do_write(
                    offset,
                    memory_obj,
                    copy_size,
                ),
            )
            if committed:
                self._invoke_on_complete_callback(key, on_complete_callback)

        return futures or None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Load one key from DAX into a CPU memory object.

        Args:
            key: Cache key to retrieve.

        Returns:
            Restored ``MemoryObj`` if the key is present and can be read;
            otherwise ``None``.

        Raises:
            RuntimeError: If CPU allocation or DAX copy fails after the key is
                reserved.
        """
        if self._is_closed_or_closing():
            return None

        reservations, _ = self._core.reserve_reads([key], prefix_only=False)
        if not reservations:
            return None

        reservation = reservations[0]
        assert self.local_cpu_backend is not None

        memory_obj: Optional[MemoryObj] = None
        touched_keys: set[CacheEngineKey] = set()
        try:
            memory_obj = self.local_cpu_backend.allocate(
                reservation.shape,
                reservation.dtype,
                reservation.fmt,
            )
            if memory_obj is None:
                return None

            self._do_read(reservation.offset, memory_obj, reservation.size)
            memory_obj.metadata.cached_positions = reservation.cached_positions
            touched_keys.add(key)
            return memory_obj
        except Exception:
            if memory_obj is not None:
                memory_obj.ref_count_down()
            raise
        finally:
            self._core.finalize_reads(reservations, touched_keys)

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Return the number of consecutive keys present in the index.

        Iterates ``keys`` in order and stops at the first miss.

        Args:
            lookup_id: Caller-supplied identifier (not used by this backend).
            keys: Ordered list of cache keys to check.
            pin: If ``True``, externally lock each found key.

        Returns:
            The count of consecutive hits from the start of ``keys``.
        """
        del lookup_id
        return self.batched_contains(keys, pin=pin)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Retrieve memory objects for consecutive keys asynchronously.

        Schedules one batched restore job on the persistent dispatch
        executor and returns only the consecutive hit prefix. Stops at the
        first key that is not found or is no longer readable.

        Args:
            lookup_id: Caller-supplied identifier (not used by this backend).
            keys: Ordered list of cache keys to retrieve.
            transfer_spec: Transfer hint (not used by this backend).

        Returns:
            A list of ``MemoryObj`` instances for the consecutive hits.
        """
        del lookup_id, transfer_spec
        if not keys or self._is_closed_or_closing():
            return []

        dispatch_executor = self._restore_dispatch_executor
        if dispatch_executor is None:
            raise RuntimeError("DaxBackend restore dispatch executor is not available")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            dispatch_executor,
            self._restore_batch,
            list(keys),
            True,
        )
        return cast(list[MemoryObj], result)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Restore a batch of DAX-backed cache entries synchronously.

        The returned list preserves the input order. Entries that are missing
        or no longer readable remain ``None`` so callers keep positional
        alignment with ``keys``.

        Args:
            keys: Ordered cache keys to restore from the DAX arena.

        Returns:
            A list aligned with ``keys`` containing restored ``MemoryObj``
            instances or ``None`` for entries that could not be read.
        """
        if not keys or self._is_closed_or_closing():
            return []

        dispatch_executor = self._restore_dispatch_executor
        if dispatch_executor is None:
            raise RuntimeError("DaxBackend restore dispatch executor is not available")

        batch_keys = list(keys)
        if threading.current_thread().name.startswith("dax-restore-dispatch"):
            return cast(
                List[Optional[MemoryObj]],
                self._restore_batch(batch_keys, False),
            )

        future = dispatch_executor.submit(self._restore_batch, batch_keys, False)
        return cast(List[Optional[MemoryObj]], future.result())

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Return the number of consecutive keys present in the index.

        Synchronous variant of :meth:`batched_async_contains`.

        Args:
            keys: Ordered list of cache keys to check.
            pin: If ``True``, externally lock each found key.

        Returns:
            The count of consecutive hits from the start of ``keys``.
        """
        hit_count = 0
        for key in keys:
            if not self._core.exists_many([key], lock=pin)[0]:
                break
            hit_count += 1
        return hit_count

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """Remove multiple keys from the backend.

        Args:
            keys: The cache keys to remove.
            force: Passed through to :meth:`remove`.

        Returns:
            The number of keys that were actually present and removed.
        """
        return sum(self._core.delete_many(keys, force=force))

    def get_allocator_backend(self) -> LocalCPUBackend:
        """Return the CPU allocator backend used for read buffers.

        Raises:
            RuntimeError: If no ``local_cpu_backend`` is available.
        """
        if self.local_cpu_backend is None:
            raise RuntimeError("DaxBackend has no allocator backend available")
        return self.local_cpu_backend

    def close(self) -> None:
        """Release restore workers, staging buffers, and DAX resources."""
        with self._close_lock:
            if self._closed:
                return
            self._closing = True

        self._release_restore_resources()
        self._core.close()

        with self._close_lock:
            self._closed = True

    @staticmethod
    def _get_positive_int_extra(
        extra_config: dict[str, Any],
        key: str,
        default: int,
    ) -> int:
        value = extra_config.get(key, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"extra_config['{key}'] must be a positive integer"
            ) from exc
        if parsed <= 0:
            raise ValueError(f"extra_config['{key}'] must be a positive integer")
        return parsed

    def _is_closed_or_closing(self) -> bool:
        with self._close_lock:
            return self._closing or self._closed

    def _release_restore_resources(self) -> None:
        dispatch_executor = self._restore_dispatch_executor
        self._restore_dispatch_executor = None
        if dispatch_executor is not None:
            dispatch_executor.shutdown(wait=True)

        restore_executor = self._restore_executor
        self._restore_executor = None
        if restore_executor is not None:
            restore_executor.shutdown(wait=True)

        if self._retrieve_staging_slab_ptr:
            try:
                lmc_ops.free_pinned_ptr(self._retrieve_staging_slab_ptr)
            except Exception as exc:
                logger.warning("Failed to free DAX retrieve slab: %s", exc)

        self._retrieve_staging_slab_ptr = 0
        self._retrieve_staging_slab_bytes = 0
        self._restore_region_bytes = 0

    def _invoke_on_complete_callback(
        self,
        key: CacheEngineKey,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        if on_complete_callback is None:
            return
        try:
            on_complete_callback(key)
        except Exception as exc:
            logger.warning("on_complete_callback failed for key %s: %s", key, exc)

    def _do_write(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(self._base_ptr + offset),
            ctypes.c_void_p(memory_obj.data_ptr),
            size,
        )

    def _do_read(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(memory_obj.data_ptr),
            ctypes.c_void_p(self._base_ptr + offset),
            size,
        )

    def _batched_memcpy(
        self,
        src_ptrs: Sequence[int],
        dst_ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        if not src_ptrs:
            return
        if hasattr(lmc_ops, "batched_memcpy"):
            lmc_ops.batched_memcpy(list(src_ptrs), list(dst_ptrs), list(sizes))
            return

        for src_ptr, dst_ptr, size in zip(src_ptrs, dst_ptrs, sizes, strict=True):
            ctypes.memmove(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                size,
            )

    def _allocate_restore_outputs(self, reserved: Sequence[_RestoreItem]) -> None:
        assert self.local_cpu_backend is not None

        grouped_items: OrderedDict[
            tuple[tuple[int, ...], torch.dtype, MemoryFormat], list[_RestoreItem]
        ] = OrderedDict()
        for item in reserved:
            grouped_items.setdefault(
                (tuple(item.shape), item.dtype, item.fmt),
                [],
            ).append(item)

        for group_items in grouped_items.values():
            first = group_items[0]
            outputs: Optional[list[MemoryObj]] = None
            if len(group_items) > 1:
                outputs = self.local_cpu_backend.batched_allocate(
                    first.shape,
                    first.dtype,
                    len(group_items),
                    first.fmt,
                )

            if outputs is None:
                outputs = []
                for _ in group_items:
                    memory_obj = self.local_cpu_backend.allocate(
                        first.shape,
                        first.dtype,
                        first.fmt,
                    )
                    if memory_obj is None:
                        for allocated in outputs:
                            allocated.ref_count_down()
                        raise RuntimeError(
                            "DaxBackend batched restore allocation failed"
                        )
                    outputs.append(memory_obj)

            for item, memory_obj in zip(group_items, outputs, strict=True):
                item.memory_obj = memory_obj

    def _build_restore_waves(
        self,
        reserved: Sequence[_RestoreItem],
    ) -> list[_RestoreWave]:
        if not reserved:
            return []

        sorted_items = sorted(reserved, key=lambda item: item.offset)
        waves: list[_RestoreWave] = []
        next_item_idx = 0

        while next_item_idx < len(sorted_items):
            regions: list[_RestoreRegion] = []
            for region_index in range(self._restore_max_regions):
                if next_item_idx >= len(sorted_items):
                    break

                region_items: list[_RestoreItem] = []
                region_spans: list[_RestoreSpan] = []
                used_bytes = 0

                while next_item_idx < len(sorted_items):
                    item = sorted_items[next_item_idx]
                    if item.size > self._restore_region_bytes:
                        raise RuntimeError(
                            f"DaxBackend restore item size {item.size} exceeds "
                            "region capacity "
                            f"{self._restore_region_bytes}"
                        )
                    if (
                        used_bytes > 0
                        and used_bytes + item.size > self._restore_region_bytes
                    ):
                        break

                    item.slab_offset = used_bytes
                    region_items.append(item)
                    if (
                        region_spans
                        and region_spans[-1].src_offset + region_spans[-1].size
                        == item.offset
                        and region_spans[-1].slab_offset + region_spans[-1].size
                        == item.slab_offset
                    ):
                        region_spans[-1].size += item.size
                    else:
                        region_spans.append(
                            _RestoreSpan(
                                src_offset=item.offset,
                                slab_offset=item.slab_offset,
                                size=item.size,
                            )
                        )

                    used_bytes += item.size
                    next_item_idx += 1

                regions.append(
                    _RestoreRegion(
                        region_index=region_index,
                        slab_offset=region_index * self._restore_region_bytes,
                        total_bytes=used_bytes,
                        items=region_items,
                        spans=region_spans,
                    )
                )

            waves.append(_RestoreWave(regions=regions))

        return waves

    def _restore_region(self, region: _RestoreRegion) -> None:
        if region.total_bytes <= 0 or not region.items:
            return
        if self._retrieve_staging_slab_ptr == 0:
            raise RuntimeError("DaxBackend retrieve slab is not allocated")

        slab_base_ptr = self._retrieve_staging_slab_ptr + region.slab_offset
        dax_src_ptrs = [self._base_ptr + span.src_offset for span in region.spans]
        slab_dst_ptrs = [slab_base_ptr + span.slab_offset for span in region.spans]
        dax_copy_sizes = [span.size for span in region.spans]
        self._batched_memcpy(dax_src_ptrs, slab_dst_ptrs, dax_copy_sizes)

        slab_src_ptrs = [slab_base_ptr + item.slab_offset for item in region.items]
        dst_ptrs = [cast(MemoryObj, item.memory_obj).data_ptr for item in region.items]
        out_sizes = [item.size for item in region.items]
        self._batched_memcpy(slab_src_ptrs, dst_ptrs, out_sizes)

    def _run_restore_waves(self, waves: Sequence[_RestoreWave]) -> None:
        restore_executor = self._restore_executor
        if restore_executor is None:
            raise RuntimeError("DaxBackend restore executor is not available")

        for wave in waves:
            futures = [
                restore_executor.submit(self._restore_region, region)
                for region in wave.regions
                if region.items
            ]
            for future in futures:
                future.result()

    def _cleanup_restore_outputs(self, reserved: Sequence[_RestoreItem]) -> None:
        for item in reserved:
            if item.memory_obj is not None:
                item.memory_obj.ref_count_down()
                item.memory_obj = None

    def _restore_batch(
        self,
        keys: list[CacheEngineKey],
        prefix_only: bool,
    ) -> list[Optional[MemoryObj]]:
        reservations, _ = self._core.reserve_reads(keys, prefix_only=prefix_only)
        results: list[Optional[MemoryObj]] = [None] * len(keys)
        if not reservations:
            return [] if prefix_only else results

        reserved_items = [
            _RestoreItem(
                result_index=reservation.result_index,
                key=reservation.key,
                offset=reservation.offset,
                size=reservation.size,
                shape=reservation.shape,
                dtype=reservation.dtype,
                fmt=reservation.fmt,
                cached_positions=reservation.cached_positions,
                slot_id=reservation.slot_id,
                generation=reservation.generation,
            )
            for reservation in reservations
        ]

        touched_keys: set[CacheEngineKey] = set()
        try:
            self._allocate_restore_outputs(reserved_items)
            waves = self._build_restore_waves(reserved_items)
            self._run_restore_waves(waves)
            for item in reserved_items:
                memory_obj = cast(MemoryObj, item.memory_obj)
                memory_obj.metadata.cached_positions = item.cached_positions
                results[item.result_index] = memory_obj
                touched_keys.add(item.key)
        except Exception:
            self._cleanup_restore_outputs(reserved_items)
            self._core.finalize_reads(reservations, set())
            raise

        self._core.finalize_reads(reservations, touched_keys)
        if prefix_only:
            prefix_results = [
                cast(MemoryObj, results[item.result_index]) for item in reserved_items
            ]
            return cast(
                list[Optional[MemoryObj]],
                prefix_results,
            )
        return results

    async def _submit_write(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        try:
            committed = await asyncio.to_thread(
                self._core.put_one,
                key,
                memory_obj,
                writer=lambda offset, obj, size: self._do_write(offset, obj, size),
            )
            if committed:
                self._invoke_on_complete_callback(key, on_complete_callback)
        finally:
            memory_obj.ref_count_down()
