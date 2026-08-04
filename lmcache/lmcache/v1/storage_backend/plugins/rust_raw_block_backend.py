# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)
from lmcache.v1.storage_backend.raw_block import (
    DEFAULT_IOURING_QUEUE_DEPTH,
    RawBlockCore,
    RawBlockCoreConfig,
    RawBlockKeySpec,
    decode_legacy_key,
    encode_legacy_key,
    normalize_raw_block_io_engine,
    round_up,
    validate_raw_block_io_options,
)

if TYPE_CHECKING:
    # Standard
    from concurrent.futures import Future

    # First Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1

TPRankKey = int | str
PerTPDevicePaths = Mapping[TPRankKey, str]


def _validate_per_tp_device_paths(per_tp_devices: PerTPDevicePaths) -> None:
    """Validate that each TP rank uses a distinct raw-block device path.

    Args:
        per_tp_devices: Mapping from TP rank to raw-block device path.

    Raises:
        ValueError: If the same device path is assigned to multiple ranks.
    """
    values = list(per_tp_devices.values())
    if len(values) != len(set(values)):
        raise ValueError(
            "Duplicate device path configured in rust_raw_block.per_tp_device_paths"
        )


def _get_per_tp_device_path(
    per_tp_devices: PerTPDevicePaths, tp_rank: int
) -> Optional[str]:
    """Return the device path configured for a TP rank.

    Args:
        per_tp_devices: Mapping with string or integer rank keys.
        tp_rank: Tensor-parallel rank to look up.

    Returns:
        The configured path, or None when the rank is absent.
    """
    return per_tp_devices.get(str(tp_rank), per_tp_devices.get(tp_rank))


class RustRawBlockBackend(StoragePluginInterface):
    """
    Legacy raw-block storage plugin wrapper.

    The durable raw-device/index/checkpoint logic now lives in RawBlockCore.
    This wrapper preserves the existing non-MP interface and prefix semantics:
    - TP>1 still uses explicit per-TP device partitions
    - batched_async_contains reports only the leading hit prefix
    - batched_get_{blocking,non_blocking} load only the leading hit prefix
    """

    def __init__(
        self,
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ):
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.loop is None:
            raise ValueError("RustRawBlockBackend requires an asyncio event loop")
        if self.local_cpu_backend is None:
            raise ValueError("RustRawBlockBackend requires local_cpu_backend")
        if self.config is None:
            raise ValueError("RustRawBlockBackend requires config")

        extra = self.config.extra_config or {}

        self.device_path: str
        if self.metadata is not None and self.metadata.world_size > 1:
            tp_rank = self.metadata.worker_id
            per_tp_devices = extra.get("rust_raw_block.per_tp_device_paths", {})
            if not isinstance(per_tp_devices, Mapping):
                raise ValueError(
                    "rust_raw_block.per_tp_device_paths must be a mapping from "
                    "TP rank to device path"
                )
            if not per_tp_devices:
                raise ValueError(
                    "For TP > 1, rust_raw_block.per_tp_device_paths is required. "
                    "Each TP worker must have an explicit device path configured."
                )
            _validate_per_tp_device_paths(per_tp_devices)
            device_path = _get_per_tp_device_path(per_tp_devices, tp_rank)
            if not device_path:
                raise ValueError(
                    f"No device path configured for TP rank {tp_rank}. "
                    f"Available ranks: {list(per_tp_devices.keys())}"
                )
            self.device_path = device_path
        else:
            self.device_path = str(extra.get("rust_raw_block.device_path", "") or "")
            if not self.device_path:
                raise ValueError(
                    "extra_config['rust_raw_block.device_path'] is required"
                )

        self._core = RawBlockCore(
            self._build_core_config(extra),
            key_namespace="legacy",
        )
        if self._core.io_engine == "io_uring":
            try:
                self._core.register_fixed_buffers_from_allocator(
                    self.local_cpu_backend.get_memory_allocator()
                )
            except Exception as e:
                logger.warning(
                    "RustRawBlockBackend: failed to register io_uring fixed "
                    "buffers: %s. Falling back to non-fixed buffer mode.",
                    e,
                )
        self._warn_if_loaded_metadata_looks_cross_rank()

        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._pin_lock = threading.Lock()
        self._pinned_keys: set[str] = set()

    def __str__(self) -> str:
        return "RustRawBlockBackend"

    @property
    def capacity_bytes(self) -> int:
        """Return the effective raw-block capacity in bytes."""
        return int(self._core.capacity_bytes)

    @property
    def block_align(self) -> int:
        """Return the configured raw-device block alignment."""
        return int(self._core.block_align)

    @property
    def header_bytes(self) -> int:
        """Return the per-slot header reservation in bytes."""
        return int(self._core.header_bytes)

    @property
    def slot_bytes(self) -> int:
        """Return the configured raw-block slot size in bytes."""
        return int(self._core.slot_bytes)

    @property
    def meta_total_bytes(self) -> int:
        """Return the reserved metadata checkpoint region size."""
        return int(self._core.meta_total_bytes)

    @property
    def meta_magic_text(self) -> str:
        """Return the ASCII metadata checkpoint magic."""
        return str(self._core.meta_magic_text)

    @property
    def meta_version(self) -> int:
        """Return the metadata checkpoint format version."""
        return int(self._core.meta_version)

    @property
    def data_base_offset(self) -> int:
        """Return the byte offset where data slots begin."""
        return self._core.data_base_offset()

    @property
    def _raw(self) -> Any:
        """Return the raw device handle for legacy test compatibility."""
        return self._core.raw_device()

    @_raw.setter
    def _raw(self, raw_device: Any) -> None:
        """Replace the raw device handle for legacy test compatibility."""
        self._core.set_raw_device_for_testing(raw_device)

    def lock_refcount(self, encoded_key: str) -> int:
        """Return the L2 lock refcount for a legacy encoded key."""
        return self._core.lock_refcount(encoded_key)

    def inflight_io_count(self) -> int:
        """Return the number of active raw-device I/O operations."""
        return self._core.inflight_io_count()

    def indexed_key_count(self) -> int:
        """Return the number of keys currently indexed by raw-block."""
        return self._core.indexed_key_count()

    def entry_offset(self, key: CacheEngineKey) -> int | None:
        """Return the raw-block slot offset for a legacy key."""
        return self._core.entry_offset(encode_legacy_key(key).encoded)

    def metadata_container_offsets(self) -> list[int]:
        """Return checkpoint metadata container offsets in bytes."""
        return self._core.metadata_container_offsets()

    def apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Validate and apply a raw-block metadata checkpoint payload."""
        return self._core.apply_loaded_state(data)

    def _build_core_config(self, extra: Mapping[str, Any]) -> RawBlockCoreConfig:
        block_align = int(extra.get("rust_raw_block.block_align", 4096))
        header_bytes = int(extra.get("rust_raw_block.header_bytes", 4096))
        use_odirect = bool(extra.get("rust_raw_block.use_odirect", False))
        enable_zero_copy = bool(extra.get("rust_raw_block.enable_zero_copy", True))
        capacity_bytes = int(extra.get("rust_raw_block.capacity_bytes", 0))
        io_engine = normalize_raw_block_io_engine(
            extra.get("rust_raw_block.io_engine"),
            use_iouring=extra.get("rust_raw_block.use_iouring"),
            use_uring=extra.get("rust_raw_block.use_uring"),
        )
        iouring_queue_depth = int(
            extra.get("rust_raw_block.iouring_queue_depth", DEFAULT_IOURING_QUEUE_DEPTH)
        )
        use_uring_cmd = bool(extra.get("rust_raw_block.use_uring_cmd", False))
        max_data_transfer_size = int(
            extra.get("rust_raw_block.max_data_transfer_size", 0)
        )
        validate_raw_block_io_options(
            iouring_queue_depth=iouring_queue_depth,
        )
        meta_total_bytes = int(
            extra.get("rust_raw_block.meta_total_bytes", 128 * 1024 * 1024)
        )
        meta_magic_raw = extra.get("rust_raw_block.meta_magic", "LMCIDX01")
        if isinstance(meta_magic_raw, str):
            meta_magic = meta_magic_raw.encode("ascii")
        elif isinstance(meta_magic_raw, bytes):
            meta_magic = meta_magic_raw
        else:
            raise ValueError("rust_raw_block.meta_magic must be str or bytes")

        get_full_chunk_size_bytes = getattr(
            self.local_cpu_backend, "get_full_chunk_size_bytes", None
        )
        if callable(get_full_chunk_size_bytes):
            full_chunk_bytes = int(get_full_chunk_size_bytes())
        else:
            get_full_chunk_size = getattr(
                self.local_cpu_backend, "get_full_chunk_size", None
            )
            if not callable(get_full_chunk_size):
                raise ValueError(
                    "local_cpu_backend must expose get_full_chunk_size_bytes() "
                    "or get_full_chunk_size()"
                )
            full_chunk_bytes = int(get_full_chunk_size())
        default_slot_bytes = round_up(header_bytes + full_chunk_bytes, block_align)
        slot_bytes = int(extra.get("rust_raw_block.slot_bytes", default_slot_bytes))

        return RawBlockCoreConfig(
            device_path=self.device_path,
            capacity_bytes=capacity_bytes,
            block_align=block_align,
            header_bytes=header_bytes,
            slot_bytes=slot_bytes,
            use_odirect=use_odirect,
            enable_zero_copy=enable_zero_copy,
            meta_total_bytes=meta_total_bytes,
            meta_magic=meta_magic,
            meta_version=int(
                extra.get("rust_raw_block.meta_version", _DEFAULT_META_VERSION)
            ),
            meta_checkpoint_interval_sec=int(
                extra.get("rust_raw_block.meta_checkpoint_interval_sec", 60)
            ),
            meta_idle_quiet_ms=int(extra.get("rust_raw_block.meta_idle_quiet_ms", 100)),
            meta_enable_periodic=bool(
                extra.get("rust_raw_block.meta_enable_periodic", True)
            ),
            load_checkpoint_on_init=bool(
                extra.get("rust_raw_block.load_checkpoint_on_init", True)
            ),
            meta_verify_on_load=bool(
                extra.get("rust_raw_block.meta_verify_on_load", True)
            ),
            max_data_transfer_size=max_data_transfer_size,
            io_engine=io_engine,
            iouring_queue_depth=iouring_queue_depth,
            use_uring_cmd=use_uring_cmd,
        )

    def _warn_if_loaded_metadata_looks_cross_rank(self) -> None:
        if self.metadata is None:
            return
        first_encoded_key = self._core.first_encoded_key()
        if first_encoded_key is None:
            return
        try:
            first_loaded_key = decode_legacy_key(first_encoded_key)
        except Exception:
            return
        expected_worker_id = int(self.metadata.worker_id)
        loaded_worker_id = int(first_loaded_key.worker_id)
        if loaded_worker_id == expected_worker_id:
            return
        logger.warning(
            "RustRawBlockBackend: loaded metadata may belong to another "
            "worker (device=%s, current_worker_id=%d, "
            "first_entry_worker_id=%d, first_entry_key=%s)",
            self.device_path,
            expected_worker_id,
            loaded_worker_id,
            first_loaded_key.to_string(),
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        spec = encode_legacy_key(key)
        return (
            self._pin_if_needed(spec.encoded)
            if pin
            else self._core.contains_key(
                spec.encoded,
                lock=False,
            )
        )

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        spec = encode_legacy_key(key)
        return self._pin_if_needed(spec.encoded)

    def unpin(self, key: CacheEngineKey) -> bool:
        spec = encode_legacy_key(key)
        return self._unpin_if_needed(spec.encoded)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        spec = encode_legacy_key(key)
        with self._pin_lock:
            removed = self._core.delete_many([spec.encoded], force=force)[0]
            if removed:
                self._pinned_keys.discard(spec.encoded)
        return removed

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,  # noqa: ARG002
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future] | None:
        del transfer_spec
        loop = self.loop
        if loop is None:
            raise RuntimeError("RustRawBlockBackend requires an asyncio event loop")

        pending: list[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]] = []
        for key, obj in zip(keys, objs, strict=False):
            with self._put_lock:
                if key in self._put_tasks:
                    continue
                self._put_tasks.add(key)

            spec = encode_legacy_key(key)
            exists = self._core.contains_key(
                spec.encoded,
                lock=False,
            ) or self._core.exists_inflight(spec.encoded)
            if exists:
                with self._put_lock:
                    self._put_tasks.discard(key)
                continue

            obj.ref_count_up()
            pending.append((key, spec, obj))

        if not pending:
            return None

        # On scheduling failure, roll back unscheduled items; scheduled ones
        # release their ref / put-task entry in their coroutine's finally.
        scheduled_count = 0
        try:
            if self._core.io_engine == "io_uring" and len(pending) > 1:
                coro = self._submit_put_many(pending, on_complete_callback)
                try:
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception:
                    coro.close()
                    raise
                return [fut]

            futures: list[Future] = []
            for key, spec, obj in pending:
                coro = self._submit_put_one(key, spec, obj, on_complete_callback)
                try:
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception:
                    coro.close()
                    raise
                futures.append(fut)
                scheduled_count += 1
            return futures
        except Exception:
            for _, _, obj in pending[scheduled_count:]:
                obj.ref_count_down()
            with self._put_lock:
                for key, _, _ in pending[scheduled_count:]:
                    self._put_tasks.discard(key)
            raise

    async def _submit_put_one(
        self,
        key: CacheEngineKey,
        spec: RawBlockKeySpec,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        try:
            put_result = await asyncio.to_thread(
                self._core.put_many,
                [spec],
                [memory_obj],
            )
            if not put_result.results or not put_result.results[0]:
                raise RuntimeError(f"Failed to persist raw-block key {spec.encoded}")
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)
        finally:
            memory_obj.ref_count_down()
            with self._put_lock:
                self._put_tasks.discard(key)

    async def _submit_put_many(
        self,
        pending: Sequence[tuple[CacheEngineKey, RawBlockKeySpec, MemoryObj]],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        """Persist multiple legacy raw-block keys in one background batch.

        Args:
            pending: Ordered ``(key, spec, memory_obj)`` tuples to persist.
            on_complete_callback: Optional per-key completion callback.

        Raises:
            RuntimeError: If any key fails to persist.
            Exception: Propagates raw-device write failures from the core.
        """
        keys = [item[0] for item in pending]
        specs = [item[1] for item in pending]
        memory_objs = [item[2] for item in pending]
        try:
            put_result = await asyncio.to_thread(
                self._core.put_many,
                specs,
                memory_objs,
            )
            if len(put_result.results) != len(pending) or not all(put_result.results):
                failed = []

                for key, spec, ok in zip(keys, specs, put_result.results, strict=False):
                    if ok:
                        if on_complete_callback is not None:
                            try:
                                on_complete_callback(key)
                            except Exception as e:
                                logger.warning(
                                    "on_complete_callback failed for key %s: %s", key, e
                                )
                    else:
                        failed.append(spec.encoded)

                if failed:
                    raise RuntimeError(
                        "Failed to persist raw-block keys: " + ", ".join(failed)
                    )
            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            "on_complete_callback failed for key %s: %s", key, e
                        )
        finally:
            for key, _spec, memory_obj in pending:
                memory_obj.ref_count_down()
                with self._put_lock:
                    self._put_tasks.discard(key)

    def _batched_get_prefix(
        self,
        keys: Sequence[CacheEngineKey],
    ) -> list[MemoryObj]:
        if not keys:
            return []

        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        allocated: list[MemoryObj] = []
        locked_specs: list[RawBlockKeySpec] = []
        with self._pin_lock:
            prefix_metas = self._core.get_metadata_prefix(
                encoded_keys,
                lock=True,
                skip_locked=self._pinned_keys,
            )
            prefix_specs = specs[: len(prefix_metas)]
            locked_specs = [
                spec for spec in prefix_specs if spec.encoded not in self._pinned_keys
            ]

        if not prefix_specs:
            return []

        try:
            for spec, meta in zip(prefix_specs, prefix_metas, strict=False):
                if meta.shape is None or meta.dtype is None:
                    logger.warning(
                        "Raw-block metadata missing shape/dtype for key %s; "
                        "aborting prefix load",
                        spec.encoded,
                    )
                    break
                if self.local_cpu_backend is None:
                    raise RuntimeError("RustRawBlockBackend requires local_cpu_backend")
                memory_obj = self.local_cpu_backend.allocate(
                    meta.shape,
                    meta.dtype,
                    meta.fmt,
                )
                if memory_obj is None:
                    logger.error("Failed to allocate memory for key %s", spec.encoded)
                    break
                allocated.append(memory_obj)

            if not allocated:
                return []

            load_specs = prefix_specs[: len(allocated)]
            load_results = self._core.load_many_into(
                [spec.encoded for spec in load_specs],
                allocated,
                raise_on_error=True,
            )
            loaded_count = 0
            for ok in load_results:
                if not ok:
                    break
                loaded_count += 1
            if loaded_count == len(allocated):
                return allocated

            for obj in allocated[loaded_count:]:
                obj.ref_count_down()
            return allocated[:loaded_count]
        except Exception:
            for obj in allocated:
                obj.ref_count_down()
            raise
        finally:
            self._core.unlock_many([spec.encoded for spec in locked_specs])

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        loaded = self._batched_get_prefix([key])
        return loaded[0] if loaded else None

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Synchronously load the leading raw-block hit prefix.

        Args:
            keys: Ordered legacy cache keys to load.

        Returns:
            A list aligned with ``keys`` containing loaded memory objects for
            the contiguous hit prefix and ``None`` for the remaining suffix.

        Raises:
            RuntimeError: If the local CPU allocator backend is unavailable.
            Exception: Propagates raw-device load failures from the core.
        """
        if not keys:
            return []
        loaded = self._batched_get_prefix(keys)
        return [*loaded, *([None] * (len(keys) - len(loaded)))]

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        del lookup_id
        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        results = self._core.exists_many(encoded_keys, lock=False)
        prefix_hits = 0
        for ok in results:
            if not ok:
                break
            prefix_hits += 1
        if pin and prefix_hits > 0:
            pinned_hits = 0
            for encoded_key in encoded_keys[:prefix_hits]:
                if not self._pin_if_needed(encoded_key):
                    break
                pinned_hits += 1
            prefix_hits = pinned_hits
        return prefix_hits

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Asynchronously load the leading raw-block hit prefix.

        Args:
            lookup_id: Lookup identifier supplied by the storage manager.
            keys: Ordered legacy cache keys to load.
            transfer_spec: Optional transfer metadata; unused by raw-block.

        Returns:
            Loaded memory objects for the contiguous hit prefix only.

        Raises:
            RuntimeError: If the local CPU allocator backend is unavailable.
            Exception: Propagates raw-device load failures from the core.
        """
        del lookup_id, transfer_spec
        return await asyncio.to_thread(self._batched_get_prefix, keys)

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        if self.local_cpu_backend is None:
            raise RuntimeError("RustRawBlockBackend requires local_cpu_backend")
        return self.local_cpu_backend

    def close(self) -> None:
        deadline = time.monotonic() + 10.0
        while True:
            with self._put_lock:
                pending = len(self._put_tasks)
            if pending == 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        self._core.close()

    def _pin_if_needed(self, encoded_key: str) -> bool:
        with self._pin_lock:
            if encoded_key in self._pinned_keys:
                return True
            if not self._core.exists_many([encoded_key], lock=True)[0]:
                return False
            self._pinned_keys.add(encoded_key)
            return True

    def _unpin_if_needed(self, encoded_key: str) -> bool:
        with self._pin_lock:
            if encoded_key in self._pinned_keys:
                self._core.unlock_many([encoded_key])
                self._pinned_keys.discard(encoded_key)
                return True
            return self._core.contains_key(encoded_key, lock=False)
