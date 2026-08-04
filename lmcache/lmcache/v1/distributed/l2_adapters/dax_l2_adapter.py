# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Optional, Protocol, cast
import os
import threading

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2ReconfigureError,
    L2ReconfigureStatus,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.dax.core import (
    DaxCore,
    DaxPutFromPtrResult,
    DaxReadReservation,
)

logger = init_logger(__name__)

DaxDeviceState = Literal[
    "active",
    "draining",
    "migrating",
    "resizing",
    "removing",
    "closed",
    "failed",
    "removed",
]
HotplugRemoveMode = Literal["migrate", "evict", "drain"]
HotplugResizeMode = Literal["migrate", "evict"]
DaxHotplugError = L2ReconfigureError
_DAX_RECONFIGURE_OPERATIONS = ["status", "add", "remove", "resize"]

_READABLE_STATES: set[DaxDeviceState] = {
    "active",
    "draining",
    "migrating",
    "resizing",
}
_WRITABLE_STATES: set[DaxDeviceState] = {"active"}
_CAPACITY_STATES: set[DaxDeviceState] = {
    "active",
    "draining",
    "migrating",
    "resizing",
    "removing",
}


class _EventNotifier(Protocol):
    def fileno(self) -> int: ...

    def notify(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DaxDeviceConfig:
    """One configured DAX arena in a DAX L2 adapter."""

    device_path: str
    max_dax_size_gb: float

    @property
    def max_dax_size_bytes(self) -> int:
        """Return this device's configured arena size in bytes."""
        return int(self.max_dax_size_gb * 1024**3)


@dataclass
class DaxDeviceEntry:
    """Runtime state for one mapped DAX device."""

    device_id: int
    device_path: str
    core: DaxCore[ObjectKey]
    max_dax_size_bytes: int
    slot_bytes: int
    state: DaxDeviceState


def _parse_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _parse_positive_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _validate_device_path(value: object, field_name: str = "device_path") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _reconfigure_device_path(payload: dict[str, object]) -> str:
    try:
        return _validate_device_path(payload.get("device_path"))
    except ValueError as exc:
        raise L2ReconfigureError(400, str(exc)) from exc


def _reconfigure_size_bytes(payload: dict[str, object]) -> int:
    value = payload.get("size_bytes")
    if isinstance(value, bool):
        raise L2ReconfigureError(400, "size_bytes must be a positive integer")
    try:
        return _parse_positive_int(value, "size_bytes")
    except ValueError as exc:
        raise L2ReconfigureError(400, str(exc)) from exc


class DaxL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the built-in MP Device-DAX L2 adapter."""

    def __init__(
        self,
        *,
        slot_bytes: int,
        devices: list[DaxDeviceConfig],
        hotplug_enabled: bool = False,
        num_store_workers: int = 1,
        num_lookup_workers: int = 1,
        num_load_workers: int = min(4, os.cpu_count() or 1),
    ) -> None:
        """Initialize a validated DAX L2 adapter config.

        Args:
            slot_bytes: Fixed slot size for each stored object.
            devices: Device mappings configured at startup. This may be empty
                only when hotplug is enabled.
            hotplug_enabled: Whether runtime add/remove/resize APIs are enabled.
            num_store_workers: Number of worker threads for store tasks.
            num_lookup_workers: Number of worker threads for lookup tasks.
            num_load_workers: Number of worker threads for load tasks.
        """
        if not devices and not hotplug_enabled:
            raise ValueError("devices may be empty only when hotplug_enabled is true")

        self.devices = list(devices)
        self.device_path = self.devices[0].device_path if self.devices else ""
        self.max_dax_size_gb = self.devices[0].max_dax_size_gb if self.devices else 0.0
        self.slot_bytes = slot_bytes
        self.hotplug_enabled = hotplug_enabled
        self.num_store_workers = num_store_workers
        self.num_lookup_workers = num_lookup_workers
        self.num_load_workers = num_load_workers

    @classmethod
    def from_dict(cls, d: dict) -> "DaxL2AdapterConfig":
        """Build a DAX L2 adapter config from CLI JSON.

        Args:
            d: Parsed ``--l2-adapter`` JSON object.

        Returns:
            A validated ``DaxL2AdapterConfig`` instance.

        Raises:
            ValueError: If a required field is missing or any numeric field
                is not positive.
        """
        slot_bytes = d.get("slot_bytes")
        if not isinstance(slot_bytes, int) or slot_bytes <= 0:
            raise ValueError("slot_bytes must be a positive integer")

        hotplug_enabled = bool(d.get("hotplug_enabled", False))

        devices: list[DaxDeviceConfig]
        if "devices" in d:
            raw_devices = d.get("devices")
            if not isinstance(raw_devices, list):
                raise ValueError("devices must be a list")
            devices = []
            for i, raw_device in enumerate(raw_devices):
                if not isinstance(raw_device, dict):
                    raise ValueError(f"devices[{i}] must be a dict")
                device_path = _validate_device_path(
                    raw_device.get("device_path"),
                    f"devices[{i}].device_path",
                )
                max_dax_size_gb = _parse_positive_float(
                    raw_device.get("max_dax_size_gb"),
                    f"devices[{i}].max_dax_size_gb",
                )
                max_dax_size_bytes = int(max_dax_size_gb * 1024**3)
                if max_dax_size_bytes // slot_bytes <= 0:
                    raise ValueError(
                        f"devices[{i}] configured DAX arena does not fit one slot"
                    )
                devices.append(
                    DaxDeviceConfig(
                        device_path=device_path,
                        max_dax_size_gb=max_dax_size_gb,
                    )
                )
        else:
            device_path = _validate_device_path(d.get("device_path"))
            max_dax_size_gb = _parse_positive_float(
                d.get("max_dax_size_gb"),
                "max_dax_size_gb",
            )
            max_dax_size_bytes = int(max_dax_size_gb * 1024**3)
            if max_dax_size_bytes // slot_bytes <= 0:
                raise ValueError("configured DAX arena does not fit even one slot")
            devices = [
                DaxDeviceConfig(
                    device_path=device_path,
                    max_dax_size_gb=max_dax_size_gb,
                )
            ]

        num_store_workers = _parse_positive_int(
            d.get("num_store_workers", 1),
            "num_store_workers",
        )
        num_lookup_workers = _parse_positive_int(
            d.get("num_lookup_workers", 1),
            "num_lookup_workers",
        )
        num_load_workers = _parse_positive_int(
            d.get("num_load_workers", min(4, os.cpu_count() or 1)),
            "num_load_workers",
        )

        return cls(
            devices=devices,
            hotplug_enabled=hotplug_enabled,
            slot_bytes=slot_bytes,
            num_store_workers=num_store_workers,
            num_lookup_workers=num_lookup_workers,
            num_load_workers=num_load_workers,
        )

    @classmethod
    def help(cls) -> str:
        """Return CLI help text for the DAX L2 adapter config.

        Returns:
            Human-readable field descriptions used by adapter config parsing.
        """
        return (
            "DAX L2 adapter config fields:\n"
            "- device_path (str): legacy single mmap-able DAX path or file path\n"
            "- max_dax_size_gb (float): legacy single-device mapped size in GiB\n"
            "- devices (list): optional multi-device entries with device_path and "
            "max_dax_size_gb\n"
            "- hotplug_enabled (bool): enables runtime /reconfigure/dax/* "
            "management APIs\n"
            "- slot_bytes (int): fixed slot size in bytes (required, >0)\n"
            "- num_store_workers (int): store worker threads (optional, default 1)\n"
            "- num_lookup_workers (int): lookup worker threads (optional, default 1)\n"
            "- num_load_workers (int): load worker threads "
            "(optional, default min(4, cpu_count))"
        )


class DaxL2Adapter(L2AdapterInterface):
    """MP L2 adapter that stores fixed-size objects in DAX mmap arenas."""

    def __init__(self, config: DaxL2AdapterConfig) -> None:
        """Initialize the DAX adapter and its stable worker pools.

        Args:
            config: Validated DAX adapter configuration.

        Raises:
            RuntimeError: If a configured DAX device cannot be opened or mapped.
            ValueError: If a mapped arena cannot fit at least one slot.
        """
        super().__init__(
            max_capacity_bytes=sum(d.max_dax_size_bytes for d in config.devices)
        )
        self._config = config

        self._device_lock = threading.RLock()
        self._devices: list[DaxDeviceEntry] = []
        self._next_device_id = 0
        self._key_to_device: dict[ObjectKey, int] = {}
        self._hotplug_enabled = config.hotplug_enabled

        # Backward-compatible test/debug handle for the first configured core.
        self._core: Optional[DaxCore[ObjectKey]] = None

        self._store_efd = cast(_EventNotifier, create_event_notifier())
        self._lookup_efd = cast(_EventNotifier, create_event_notifier())
        self._load_efd = cast(_EventNotifier, create_event_notifier())

        self._store_executor: Optional[ThreadPoolExecutor] = None
        self._lookup_executor: Optional[ThreadPoolExecutor] = None
        self._load_executor: Optional[ThreadPoolExecutor] = None
        self._copy_executor: Optional[ThreadPoolExecutor] = None

        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._inflight_store_tasks = 0
        self._inflight_lookup_tasks = 0
        self._inflight_load_tasks = 0

        try:
            with self._device_lock:
                for device_config in config.devices:
                    self._add_device_entry_locked(
                        device_path=device_config.device_path,
                        size_bytes=device_config.max_dax_size_bytes,
                    )
            self._store_executor = ThreadPoolExecutor(
                max_workers=config.num_store_workers,
                thread_name_prefix="dax-l2-store",
            )
            self._lookup_executor = ThreadPoolExecutor(
                max_workers=config.num_lookup_workers,
                thread_name_prefix="dax-l2-lookup",
            )
            self._load_executor = ThreadPoolExecutor(
                max_workers=config.num_load_workers,
                thread_name_prefix="dax-l2-load",
            )
            self._copy_executor = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="dax-l2-copy",
            )
        except Exception:
            self.close()
            raise

    def get_store_event_fd(self) -> int:
        """Return the pollable fd signaled when store tasks complete.

        Returns:
            File descriptor owned by this adapter's store notifier.
        """
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the pollable fd signaled when lookup tasks complete.

        Returns:
            File descriptor owned by this adapter's lookup notifier.
        """
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the pollable fd signaled when load tasks complete.

        Returns:
            File descriptor owned by this adapter's load notifier.
        """
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous L1-to-DAX store task.

        Args:
            keys: Object keys to store.
            objects: Caller-owned memory objects containing the payloads.

        Returns:
            Adapter-local task id for the submitted store task.

        Raises:
            ValueError: If ``keys`` and ``objects`` have different lengths.
            RuntimeError: If the adapter is closing or already closed.
        """
        if len(keys) != len(objects):
            raise ValueError(
                "keys and objects must have the same length, "
                f"got {len(keys)} and {len(objects)}"
            )

        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_store_tasks += 1

        assert self._store_executor is not None
        self._store_executor.submit(
            self._execute_store_task,
            task_id,
            list(keys),
            list(objects),
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Drain completed store task results.

        Returns:
            Mapping from task id to store result. Each task appears at
            most once; subsequent calls do not return already drained tasks.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        """Submit an asynchronous lookup-and-lock task.

        Found keys have their DAX external lock refcount incremented until
        ``submit_unlock`` is called.

        Args:
            keys: Object keys to look up.

        Returns:
            Adapter-local task id for the submitted lookup task.

        Raises:
            RuntimeError: If the adapter is closing or already closed.
        """
        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_lookup_tasks += 1

        assert self._lookup_executor is not None
        self._lookup_executor.submit(self._execute_lookup_task, task_id, list(keys))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Return and remove a completed lookup result.

        Args:
            task_id: Adapter-local lookup task id.

        Returns:
            A bitmap with bits set for keys that were found and locked, or
            ``None`` if the task is still pending or was already queried.
        """
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release DAX external locks acquired by lookup tasks.

        Args:
            keys: Keys whose lock refcount should be decremented.
        """
        with self._device_lock:
            cores = [entry.core for entry in self._devices]
        for core in cores:
            core.unlock_many(keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit an asynchronous DAX-to-L1 load task.

        Args:
            keys: Object keys to load.
            objects: Caller-owned destination buffers. Loaded bytes are
                written directly into these ``MemoryObj`` instances.

        Returns:
            Adapter-local task id for the submitted load task.

        Raises:
            ValueError: If ``keys`` and ``objects`` have different lengths.
            RuntimeError: If the adapter is closing or already closed.
        """
        if len(keys) != len(objects):
            raise ValueError(
                "keys and objects must have the same length, "
                f"got {len(keys)} and {len(objects)}"
            )

        with self._lock:
            self._ensure_open_locked()
            task_id = self._get_next_task_id_locked()
            self._inflight_load_tasks += 1

        assert self._load_executor is not None
        self._load_executor.submit(
            self._execute_load_task,
            task_id,
            list(keys),
            list(objects),
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        """Return and remove a completed load result.

        Args:
            task_id: Adapter-local load task id.

        Returns:
            A bitmap with bits set for keys that were successfully loaded, or
            ``None`` if the task is still pending or was already queried.
        """
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete unlocked keys from every DAX device index.

        Externally locked keys are skipped. Slots borrowed by active reads are
        reclaimed after the read finalizes.

        Args:
            keys: Object keys to delete.
        """
        if not keys or self._closed:
            return

        deleted_keys: list[ObjectKey] = []
        with self._device_lock:
            for entry in self._devices:
                deleted = entry.core.delete_many(keys, force=False)
                for key, ok in zip(keys, deleted, strict=True):
                    if ok:
                        self._key_to_device.pop(key, None)
                        if key not in deleted_keys:
                            deleted_keys.append(key)

        if deleted_keys:
            self._notify_keys_deleted(
                deleted_keys,
                [self._config.slot_bytes] * len(deleted_keys),
            )

    def get_usage(self) -> AdapterUsage:
        """Return slot-based capacity usage for this adapter.

        Returns:
            Adapter usage where ``total_bytes_used`` is derived from occupied
            DAX slots rather than payload bytes.
        """
        with self._device_lock:
            total_capacity_bytes = 0
            total_bytes_used = 0
            for entry in self._devices:
                if entry.state not in _CAPACITY_STATES:
                    continue
                status = entry.core.report_status()
                total_capacity_bytes += entry.slot_bytes * int(status["max_slots"])
                total_bytes_used += entry.slot_bytes * int(status["live_slot_count"])

        base_usage = super().get_usage()
        return AdapterUsage(
            total_bytes_used=total_bytes_used,
            total_capacity_bytes=total_capacity_bytes,
            bytes_by_cache_salt=MappingProxyType(dict(base_usage.bytes_by_cache_salt)),
        )

    def close(self) -> None:
        """Stop worker pools and release DAX resources.

        The call waits for already submitted worker tasks to finish before
        closing mapped DAX cores and event notifiers.
        """
        store_executor = None
        lookup_executor = None
        load_executor = None

        with self._lock:
            if self._closed:
                return
            self._closing = True
            store_executor = self._store_executor
            lookup_executor = self._lookup_executor
            load_executor = self._load_executor
            copy_executor = self._copy_executor
            self._store_executor = None
            self._lookup_executor = None
            self._load_executor = None
            self._copy_executor = None

        if store_executor is not None:
            store_executor.shutdown(wait=True)
        if lookup_executor is not None:
            lookup_executor.shutdown(wait=True)
        if load_executor is not None:
            load_executor.shutdown(wait=True)
        if copy_executor is not None:
            copy_executor.shutdown(wait=True)

        with self._device_lock:
            entries = list(self._devices)
            for entry in entries:
                entry.core.close()
                entry.state = "closed"
            self._key_to_device.clear()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

        with self._lock:
            self._closed = True

    def report_status(self) -> dict:
        """Return a health and capacity snapshot for this adapter.

        Returns:
            Dictionary containing health, DAX capacity, slot occupancy,
            lock/borrow counts, in-flight task counts, and restart-recovery
            capability.
        """
        hotplug = self.hotplug_status()
        with self._lock:
            closing = self._closing or self._closed

        devices = hotplug["devices"]
        return {
            "is_healthy": all(d["is_healthy"] for d in devices) and not self._closed,
            "type": "dax",
            "device_path": self._config.device_path,
            "max_dax_size_bytes": hotplug["total_capacity_bytes"],
            "slot_bytes": self._config.slot_bytes,
            "live_slot_count": sum(int(d["live_slot_count"]) for d in devices),
            "locked_key_count": sum(int(d["locked_key_count"]) for d in devices),
            "borrowed_slot_count": sum(int(d["borrowed_slot_count"]) for d in devices),
            "inflight_store_tasks": self._inflight_store_tasks,
            "inflight_lookup_tasks": self._inflight_lookup_tasks,
            "inflight_load_tasks": self._inflight_load_tasks,
            "closing": closing,
            "supports_restart_recovery": False,
            "hotplug_enabled": self._hotplug_enabled,
            "devices": devices,
            "num_devices": len(devices),
        }

    def hotplug_status(self) -> dict:
        """Return runtime status for this DAX adapter.

        Returns:
            JSON-serializable status for the stable adapter facade and every
            mapped DAX device.
        """
        with self._device_lock:
            devices = [
                self._device_status_locked(index, entry)
                for index, entry in enumerate(self._devices)
            ]
            total_capacity_bytes = sum(
                int(d["max_slots"]) * int(d["slot_bytes"])
                for d in devices
                if d["state"] in _CAPACITY_STATES
            )
            total_used_bytes = sum(
                int(d["live_slot_count"]) * int(d["slot_bytes"])
                for d in devices
                if d["state"] in _CAPACITY_STATES
            )
            return {
                "hotplug_enabled": self._hotplug_enabled,
                "slot_bytes": self._config.slot_bytes,
                "total_capacity_bytes": total_capacity_bytes,
                "total_used_bytes": total_used_bytes,
                "devices": devices,
            }

    def reconfigure_status(self) -> L2ReconfigureStatus:
        """Return generic runtime reconfiguration status for this adapter.

        Returns:
            Standard reconfiguration status with DAX hotplug details nested
            under ``status``.
        """
        return {
            "backend": "dax",
            "supported_operations": list(_DAX_RECONFIGURE_OPERATIONS),
            "status": self.hotplug_status(),
        }

    def reconfigure(
        self,
        operation: str,
        payload: dict[str, object],
    ) -> dict:
        """Apply one generic runtime reconfiguration operation.

        Args:
            operation: One of ``status``, ``add``, ``remove``, or ``resize``.
            payload: DAX-specific operation payload.

        Returns:
            JSON-serializable operation result.

        Raises:
            L2ReconfigureError: If the operation or payload is invalid, or if the
                underlying DAX hotplug operation fails.
        """
        if operation == "status":
            return dict(self.reconfigure_status())

        if operation == "add":
            device_path = _reconfigure_device_path(payload)
            size_bytes = _reconfigure_size_bytes(payload)
            return self.hotplug_add_device(device_path, size_bytes)

        if operation == "remove":
            device_path = _reconfigure_device_path(payload)
            mode = payload.get("mode", "migrate")
            if not isinstance(mode, str):
                raise L2ReconfigureError(400, "mode must be migrate, evict, or drain")
            force = payload.get("force", False)
            if not isinstance(force, bool):
                raise L2ReconfigureError(400, "force must be a boolean")
            return self.hotplug_remove_device(
                device_path,
                cast(HotplugRemoveMode, mode),
                force,
            )

        if operation == "resize":
            device_path = _reconfigure_device_path(payload)
            size_bytes = _reconfigure_size_bytes(payload)
            mode = payload.get("mode", "migrate")
            if not isinstance(mode, str):
                raise L2ReconfigureError(400, "mode must be migrate or evict")
            force = payload.get("force", False)
            if not isinstance(force, bool):
                raise L2ReconfigureError(400, "force must be a boolean")
            return self.hotplug_resize_device(
                device_path,
                size_bytes,
                cast(HotplugResizeMode, mode),
                force,
            )

        raise L2ReconfigureError(
            400,
            f"unsupported DAX reconfigure operation: {operation}",
        )

    def hotplug_add_device(self, device_path: str, size_bytes: int) -> dict:
        """Map and activate one additional DAX device.

        Args:
            device_path: Path to an existing readable and writable DAX device.
            size_bytes: Number of bytes to map.

        Returns:
            JSON-serializable operation result.

        Raises:
            L2ReconfigureError: If hotplug is disabled or the request is invalid.
        """
        self._ensure_hotplug_enabled()
        device_path = device_path.strip()
        if not device_path:
            raise L2ReconfigureError(400, "device_path must be non-empty")
        if size_bytes <= 0:
            raise L2ReconfigureError(400, "size_bytes must be > 0")
        if size_bytes // self._config.slot_bytes <= 0:
            raise L2ReconfigureError(400, "size_bytes does not fit one slot")

        with self._device_lock:
            for index, entry in enumerate(self._devices):
                if entry.device_path != device_path or entry.state in {
                    "closed",
                    "removed",
                    "failed",
                }:
                    continue
                if entry.max_dax_size_bytes == size_bytes:
                    return {
                        "status": "ok",
                        "operation": "add",
                        "adapter_index": 0,
                        "device": self._device_status_locked(index, entry),
                    }
                raise L2ReconfigureError(
                    409,
                    "device_path already active with a different size",
                )

            try:
                entry = self._add_device_entry_locked(
                    device_path=device_path,
                    size_bytes=size_bytes,
                )
            except ValueError as exc:
                logger.exception("Invalid DAX hotplug add request")
                raise L2ReconfigureError(
                    400,
                    "invalid DAX hotplug add request",
                ) from exc
            except RuntimeError as exc:
                logger.exception("Failed to map DAX hotplug device")
                raise L2ReconfigureError(400, "failed to map DAX device") from exc

            index = self._devices.index(entry)
            return {
                "status": "ok",
                "operation": "add",
                "adapter_index": 0,
                "device": self._device_status_locked(index, entry),
            }

    def hotplug_remove_device(
        self,
        device_path: str,
        mode: HotplugRemoveMode,
        force: bool = False,
    ) -> dict:
        """Remove or drain a DAX device.

        Args:
            device_path: Device path to remove.
            mode: ``migrate`` preserves KV on another device, ``evict`` deletes
                DAX-resident entries, and ``drain`` only stops new writes.
            force: Whether destructive delete may remove locked entries.

        Returns:
            JSON-serializable operation result.

        Raises:
            L2ReconfigureError: If the request is invalid, blocked, or lacks
                destination capacity.
        """
        self._ensure_hotplug_enabled()
        self._validate_remove_mode(mode)

        with self._device_lock:
            source_index, source = self._get_device_for_path_locked(device_path)
            if mode == "drain":
                source.state = "draining"
                return {
                    "status": "ok",
                    "operation": "drain",
                    "adapter_index": 0,
                    "device_path": source.device_path,
                    "index": source_index,
                    "state": source.state,
                }

            old_state = source.state
            source.state = "draining"
            try:
                blocked = self._blocked_payload_locked(source)
                if blocked is not None and not force:
                    raise L2ReconfigureError(
                        409,
                        "device has locked or borrowed slots",
                        payload=blocked,
                    )

                keys = source.core.snapshot_keys()
                if mode == "migrate":
                    self._assert_no_destination_duplicates_locked(source, keys)
                    moved_keys, moved_bytes = self._migrate_keys_locked(source, keys)
                    self._delete_source_keys_after_migration_locked(
                        source,
                        moved_keys,
                        force,
                    )
                    deleted_keys = []
                    source_slots_freed = len(moved_keys)
                else:
                    deleted_keys = self._evict_keys_locked(source, keys, force=force)
                    moved_keys = []
                    moved_bytes = 0
                    source_slots_freed = len(deleted_keys)
                    if deleted_keys:
                        self._notify_keys_deleted(
                            deleted_keys,
                            [self._config.slot_bytes] * len(deleted_keys),
                        )

                source.state = "removing"
                source.core.close()
                source.state = "removed"
                self._recalculate_capacity_locked()
            except Exception:
                if source.state not in {"closed", "removed"}:
                    source.state = old_state
                raise

            return {
                "status": "ok",
                "operation": "remove",
                "adapter_index": 0,
                "device_path": source.device_path,
                "index": source_index,
                "moved_keys": len(moved_keys),
                "moved_bytes": moved_bytes,
                "deleted_keys": len(deleted_keys),
                "source_slots_freed": source_slots_freed,
                "state": "removed",
            }

    def hotplug_resize_device(
        self,
        device_path: str,
        size_bytes: int,
        mode: HotplugResizeMode,
        force: bool = False,
    ) -> dict:
        """Resize a mapped DAX device.

        Args:
            device_path: Device path to resize.
            size_bytes: New mapped byte size.
            mode: Migration behavior for shrink operations.
            force: Whether destructive shrink may evict out-of-range entries.

        Returns:
            JSON-serializable operation result.

        Raises:
            L2ReconfigureError: If the request is invalid, blocked, or lacks
                destination capacity.
        """
        self._ensure_hotplug_enabled()
        self._validate_resize_mode(mode)
        if size_bytes <= 0:
            raise L2ReconfigureError(400, "size_bytes must be > 0")
        new_max_slots = size_bytes // self._config.slot_bytes
        if new_max_slots <= 0:
            raise L2ReconfigureError(400, "size_bytes does not fit one slot")

        with self._device_lock:
            device_index, entry = self._get_device_for_path_locked(device_path)
            old_size_bytes = entry.max_dax_size_bytes
            if old_size_bytes == size_bytes:
                return {
                    "status": "ok",
                    "operation": "resize",
                    "adapter_index": 0,
                    "device_path": entry.device_path,
                    "index": device_index,
                    "old_size_bytes": old_size_bytes,
                    "new_size_bytes": size_bytes,
                    "state": entry.state,
                }

            old_state = entry.state
            entry.state = "resizing"
            try:
                if size_bytes > old_size_bytes:
                    entry.core.remap(size_bytes)
                else:
                    self._shrink_device_locked(entry, new_max_slots, mode, force)
                    entry.core.remap(size_bytes)
                entry.max_dax_size_bytes = size_bytes
                entry.state = "active" if old_state == "active" else old_state
                self._recalculate_capacity_locked()
            except L2ReconfigureError:
                entry.state = old_state
                raise
            except ValueError as exc:
                entry.state = old_state
                logger.exception("Invalid DAX hotplug resize request")
                raise L2ReconfigureError(
                    409,
                    "invalid DAX hotplug resize request",
                ) from exc
            except RuntimeError as exc:
                entry.state = old_state
                logger.exception("Failed to remap DAX hotplug device")
                raise L2ReconfigureError(400, "failed to remap DAX device") from exc

            return {
                "status": "ok",
                "operation": "resize",
                "adapter_index": 0,
                "device_path": entry.device_path,
                "index": device_index,
                "old_size_bytes": old_size_bytes,
                "new_size_bytes": size_bytes,
                "state": entry.state,
            }

    def _ensure_open_locked(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("DaxL2Adapter is closing")

    def _get_next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _signal_eventfd(self, notifier: _EventNotifier) -> None:
        try:
            notifier.notify()
        except OSError:
            logger.debug("Skipping eventfd write during adapter shutdown")

    def _execute_store_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        per_key_results: list[bool] = []
        stored_keys: list[ObjectKey] = []

        try:
            for key, obj in zip(keys, objects, strict=True):
                try:
                    with self._device_lock:
                        ok = self._store_one_locked(key, obj)
                except Exception:
                    logger.exception("DAX L2 store failed for key %s", key)
                    ok = False
                per_key_results.append(ok)
                if ok:
                    stored_keys.append(key)
        finally:
            bytes_transferred = self._config.slot_bytes * len(stored_keys)
            with self._lock:
                self._completed_store_tasks[task_id] = L2StoreResult(
                    all(per_key_results), bytes_transferred
                )
                self._inflight_store_tasks -= 1

            if stored_keys:
                self._notify_keys_stored(
                    stored_keys,
                    [self._config.slot_bytes] * len(stored_keys),
                )
            self._signal_eventfd(self._store_efd)

    def _execute_lookup_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
    ) -> None:
        bitmap = Bitmap(len(keys))
        try:
            with self._device_lock:
                remaining_indices: list[int] = []
                skip_device_by_index: dict[int, int] = {}
                mapped_indices_by_device: dict[int, list[int]] = {}

                for i, key in enumerate(keys):
                    mapped_entry = self._get_mapped_device_locked(key)
                    if (
                        mapped_entry is not None
                        and mapped_entry.state in _READABLE_STATES
                    ):
                        mapped_indices_by_device.setdefault(
                            mapped_entry.device_id,
                            [],
                        ).append(i)
                    else:
                        remaining_indices.append(i)

                readable_entries = [
                    entry for entry in self._devices if entry.state in _READABLE_STATES
                ]
                entry_by_id = {entry.device_id: entry for entry in readable_entries}

                for device_id, indices in mapped_indices_by_device.items():
                    entry = entry_by_id.get(device_id)
                    if entry is None:
                        remaining_indices.extend(indices)
                        continue

                    hits = entry.core.exists_many([keys[i] for i in indices], lock=True)
                    for i, hit in zip(indices, hits, strict=True):
                        if hit:
                            bitmap.set(i)
                            self._key_to_device[keys[i]] = entry.device_id
                        else:
                            self._key_to_device.pop(keys[i], None)
                            skip_device_by_index[i] = entry.device_id
                            remaining_indices.append(i)

                for entry in readable_entries:
                    indices = [
                        i
                        for i in remaining_indices
                        if not bitmap.test(i)
                        and skip_device_by_index.get(i) != entry.device_id
                    ]
                    if not indices:
                        continue

                    hits = entry.core.exists_many([keys[i] for i in indices], lock=True)
                    for i, hit in zip(indices, hits, strict=True):
                        if hit:
                            bitmap.set(i)
                            self._key_to_device[keys[i]] = entry.device_id
        except Exception:
            logger.exception("DAX L2 lookup failed")
        finally:
            with self._lock:
                self._completed_lookup_tasks[task_id] = bitmap
                self._inflight_lookup_tasks -= 1
            self._signal_eventfd(self._lookup_efd)

    def _execute_load_task(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        bitmap = Bitmap(len(keys))
        loaded_keys: list[ObjectKey] = []

        try:
            # coalesced DAX load path: group by device under the lock, copy
            # outside it (device cores take their own reservation locks), and
            # parallelize sub-batches across an inline pool — ctypes.memmove
            # releases the GIL, so copies genuinely overlap.
            groups: dict[int, tuple[DaxDeviceEntry, list[int]]] = {}
            with self._device_lock:
                for i, key in enumerate(keys):
                    entry = self._find_device_for_key_locked(key, lock=False)
                    if entry is None:
                        continue
                    groups.setdefault(entry.device_id, (entry, []))[1].append(i)

            sub_batches: list[tuple[DaxDeviceEntry, list[int]]] = []
            _SUB = 8  # chunks per copy task (~0.5 GiB at 64 MiB chunks)
            for entry, idxs in groups.values():
                for s in range(0, len(idxs), _SUB):
                    sub_batches.append((entry, idxs[s : s + _SUB]))

            def _load_sub(
                batch: tuple[DaxDeviceEntry, list[int]],
            ) -> tuple[list[int], list[bool]]:
                entry, idxs = batch
                gkeys = [keys[i] for i in idxs]
                gobjs = [objects[i] for i in idxs]
                try:
                    return idxs, entry.core.load_many_into(gkeys, gobjs)
                except Exception:
                    logger.exception("Sub-batch DAX load failed")
                    return idxs, [False] * len(idxs)

            copy_pool = self._copy_executor
            if len(sub_batches) <= 1 or copy_pool is None:
                results = [_load_sub(b) for b in sub_batches]
            else:
                results = list(copy_pool.map(_load_sub, sub_batches))

            for idxs, flags in results:
                for i, ok in zip(idxs, flags, strict=True):
                    if ok:
                        bitmap.set(i)
                        loaded_keys.append(keys[i])
        except Exception:
            logger.exception("DAX L2 load failed")
        finally:
            with self._lock:
                self._completed_load_tasks[task_id] = bitmap
                self._inflight_load_tasks -= 1

            if loaded_keys:
                self._notify_keys_accessed(loaded_keys)
            self._signal_eventfd(self._load_efd)

    def _store_one_locked(self, key: ObjectKey, obj: MemoryObj) -> bool:
        mapped_entry = self._get_mapped_device_locked(key)
        if mapped_entry is not None and mapped_entry.state in _WRITABLE_STATES:
            ok = mapped_entry.core.put_many([key], [obj])[0]
            if ok:
                self._key_to_device[key] = mapped_entry.device_id
                return True

        for entry in self._writable_devices_by_usage_locked(exclude_device_id=None):
            if entry is mapped_entry:
                continue
            ok = entry.core.put_many([key], [obj])[0]
            if ok:
                self._key_to_device[key] = entry.device_id
                return True
        return False

    def _find_device_for_key_locked(
        self,
        key: ObjectKey,
        *,
        lock: bool,
    ) -> Optional[DaxDeviceEntry]:
        mapped_entry = self._get_mapped_device_locked(key)
        if mapped_entry is not None and mapped_entry.state in _READABLE_STATES:
            if mapped_entry.core.exists_many([key], lock=lock)[0]:
                return mapped_entry
            self._key_to_device.pop(key, None)

        for entry in self._devices:
            if entry is mapped_entry or entry.state not in _READABLE_STATES:
                continue
            if entry.core.exists_many([key], lock=lock)[0]:
                self._key_to_device[key] = entry.device_id
                return entry
        return None

    def _get_mapped_device_locked(self, key: ObjectKey) -> Optional[DaxDeviceEntry]:
        device_id = self._key_to_device.get(key)
        if device_id is None:
            return None
        for entry in self._devices:
            if entry.device_id == device_id:
                return entry
        self._key_to_device.pop(key, None)
        return None

    def _writable_devices_by_usage_locked(
        self,
        *,
        exclude_device_id: Optional[int],
    ) -> list[DaxDeviceEntry]:
        devices = [
            entry
            for entry in self._devices
            if entry.state in _WRITABLE_STATES and entry.device_id != exclude_device_id
        ]
        return sorted(devices, key=self._device_usage_ratio_locked)

    def _device_usage_ratio_locked(self, entry: DaxDeviceEntry) -> float:
        status = entry.core.report_status()
        max_slots = int(status["max_slots"])
        if max_slots <= 0:
            return 1.0
        return int(status["live_slot_count"]) / max_slots

    def _add_device_entry_locked(
        self,
        *,
        device_path: str,
        size_bytes: int,
    ) -> DaxDeviceEntry:
        core = DaxCore[ObjectKey](
            device_path=device_path,
            max_dax_size_bytes=size_bytes,
            slot_bytes=self._config.slot_bytes,
        )
        entry = DaxDeviceEntry(
            device_id=self._next_device_id,
            device_path=device_path,
            core=core,
            max_dax_size_bytes=size_bytes,
            slot_bytes=self._config.slot_bytes,
            state="active",
        )
        self._next_device_id += 1
        self._devices.append(entry)
        if self._core is None:
            self._core = core
        self._recalculate_capacity_locked()
        return entry

    def _device_status_locked(self, index: int, entry: DaxDeviceEntry) -> dict:
        status = entry.core.report_status()
        with self._lock:
            inflight_store = self._inflight_store_tasks
            inflight_lookup = self._inflight_lookup_tasks
            inflight_load = self._inflight_load_tasks
        return {
            **status,
            "index": index,
            "device_id": entry.device_id,
            "device_path": entry.device_path,
            "state": entry.state,
            "max_dax_size_bytes": entry.max_dax_size_bytes,
            "slot_bytes": entry.slot_bytes,
            "inflight_store_tasks": inflight_store,
            "inflight_lookup_tasks": inflight_lookup,
            "inflight_load_tasks": inflight_load,
            "supports_restart_recovery": False,
        }

    def _get_device_for_path_locked(
        self,
        device_path: str,
    ) -> tuple[int, DaxDeviceEntry]:
        normalized_path = device_path.strip()
        if not normalized_path:
            raise L2ReconfigureError(400, "device_path must be non-empty")

        matches: list[tuple[int, DaxDeviceEntry]] = []
        for index, entry in enumerate(self._devices):
            if entry.device_path != normalized_path:
                continue
            if entry.state in {"closed", "removed", "failed"}:
                continue
            matches.append((index, entry))

        if len(matches) > 1:
            raise L2ReconfigureError(
                409,
                "multiple active DAX devices have the same device_path",
                payload={
                    "error": "multiple active DAX devices have the same device_path",
                    "device_path": normalized_path,
                    "matches": [
                        {
                            "index": index,
                            "device_id": entry.device_id,
                            "state": entry.state,
                        }
                        for index, entry in matches
                    ],
                },
            )

        if matches:
            return matches[0]

        raise L2ReconfigureError(404, "DAX device not found")

    def _blocked_payload_locked(
        self,
        entry: DaxDeviceEntry,
    ) -> Optional[dict[str, object]]:
        status = entry.core.report_status()
        locked_key_count = int(status["locked_key_count"])
        borrowed_slot_count = int(status["borrowed_slot_count"])
        if locked_key_count == 0 and borrowed_slot_count == 0:
            return None
        return {
            "status": "blocked",
            "reason": "device has externally locked or borrowed slots",
            "locked_key_count": locked_key_count,
            "borrowed_slot_count": borrowed_slot_count,
        }

    def _migrate_keys_locked(
        self,
        source: DaxDeviceEntry,
        keys: list[ObjectKey],
    ) -> tuple[list[ObjectKey], int]:
        if not keys:
            return [], 0

        if not self._writable_devices_by_usage_locked(
            exclude_device_id=source.device_id
        ):
            raise L2ReconfigureError(507, "no active destination DAX capacity")

        reservations = source.core.reserve_reads_for_keys(keys)
        moved_keys: list[ObjectKey] = []
        moved_bytes = 0
        touched_keys: set[ObjectKey] = set()
        try:
            for reservation in reservations:
                if self._copy_reservation_to_target_locked(source, reservation):
                    moved_keys.append(reservation.key)
                    moved_bytes += reservation.size
                    touched_keys.add(reservation.key)
        finally:
            source.core.finalize_reads(reservations, touched_keys)

        if len(moved_keys) != len(keys):
            raise L2ReconfigureError(507, "insufficient destination DAX capacity")
        return moved_keys, moved_bytes

    def _assert_no_destination_duplicates_locked(
        self,
        source: DaxDeviceEntry,
        keys: list[ObjectKey],
    ) -> None:
        if not keys:
            return

        key_set = set(keys)
        for entry in self._devices:
            if entry is source or entry.state != "active":
                continue

            duplicates = key_set.intersection(entry.core.snapshot_keys())
            if not duplicates:
                continue

            sample = list(duplicates)[:8]
            raise L2ReconfigureError(
                409,
                "migration target already contains source keys",
                payload={
                    "error": "migration target already contains source keys",
                    "device_path": entry.device_path,
                    "device_id": entry.device_id,
                    "duplicate_count": len(duplicates),
                    "sample_keys": [str(key) for key in sample],
                },
            )

    def _copy_reservation_to_target_locked(
        self,
        source: DaxDeviceEntry,
        reservation: DaxReadReservation[ObjectKey],
    ) -> bool:
        for target in self._writable_devices_by_usage_locked(
            exclude_device_id=source.device_id
        ):
            result = target.core.put_reserved_from_ptr(
                reservation.key,
                source.core.base_ptr + reservation.offset,
                reservation.size,
                reservation.shape,
                reservation.dtype,
                reservation.fmt,
                reservation.cached_positions,
            )
            if result is DaxPutFromPtrResult.INSERTED:
                self._key_to_device[reservation.key] = target.device_id
                return True
            if result in {
                DaxPutFromPtrResult.ALREADY_EXISTS,
                DaxPutFromPtrResult.INFLIGHT,
            }:
                raise L2ReconfigureError(
                    409,
                    "migration destination already contains key",
                    payload={
                        "error": "migration destination already contains key",
                        "key": str(reservation.key),
                        "device_path": target.device_path,
                        "result": result.name,
                    },
                )
            if result is not DaxPutFromPtrResult.NO_SPACE:
                raise L2ReconfigureError(
                    409,
                    "migration destination copy failed",
                    payload={
                        "error": "migration destination copy failed",
                        "key": str(reservation.key),
                        "device_path": target.device_path,
                        "result": result.name,
                    },
                )
        return False

    def _delete_source_keys_after_migration_locked(
        self,
        source: DaxDeviceEntry,
        keys: list[ObjectKey],
        force: bool,
    ) -> None:
        deleted = source.core.delete_many(keys, force=force)
        failed = [key for key, ok in zip(keys, deleted, strict=True) if not ok]
        if failed:
            raise L2ReconfigureError(
                409,
                "source device still has locked migrated keys",
                payload={
                    "status": "blocked",
                    "reason": "source device still has locked migrated keys",
                    "locked_key_count": len(failed),
                    "borrowed_slot_count": 0,
                },
            )

    def _evict_keys_locked(
        self,
        source: DaxDeviceEntry,
        keys: list[ObjectKey],
        *,
        force: bool,
    ) -> list[ObjectKey]:
        deleted = source.core.delete_many(keys, force=force)
        deleted_keys: list[ObjectKey] = []
        failed_keys: list[ObjectKey] = []
        for key, ok in zip(keys, deleted, strict=True):
            if ok:
                deleted_keys.append(key)
                if self._key_to_device.get(key) == source.device_id:
                    self._key_to_device.pop(key, None)
            else:
                failed_keys.append(key)

        if failed_keys:
            raise L2ReconfigureError(
                409,
                "device has externally locked or borrowed slots",
                payload={
                    "status": "blocked",
                    "reason": "device has externally locked or borrowed slots",
                    "locked_key_count": len(failed_keys),
                    "borrowed_slot_count": 0,
                },
            )
        return deleted_keys

    def _shrink_device_locked(
        self,
        entry: DaxDeviceEntry,
        new_max_slots: int,
        mode: HotplugResizeMode,
        force: bool,
    ) -> None:
        blocked = self._blocked_payload_locked(entry)
        if blocked is not None and not force:
            raise L2ReconfigureError(
                409,
                "device has locked or borrowed slots",
                payload=blocked,
            )
        if entry.core.can_shrink_to(new_max_slots):
            return
        keys = entry.core.snapshot_keys()
        reservations = entry.core.reserve_reads_for_keys(keys)
        high_reservations = [
            reservation
            for reservation in reservations
            if reservation.slot_id >= new_max_slots
        ]
        touched_keys: set[ObjectKey] = set()
        try:
            if mode == "migrate":
                if high_reservations and not self._writable_devices_by_usage_locked(
                    exclude_device_id=entry.device_id
                ):
                    raise L2ReconfigureError(507, "no active destination DAX capacity")
                self._assert_no_destination_duplicates_locked(
                    entry,
                    [reservation.key for reservation in high_reservations],
                )
                moved_keys: list[ObjectKey] = []
                for reservation in high_reservations:
                    if not self._copy_reservation_to_target_locked(entry, reservation):
                        raise L2ReconfigureError(
                            507,
                            "insufficient destination DAX capacity",
                        )
                    moved_keys.append(reservation.key)
                    touched_keys.add(reservation.key)
                self._delete_source_keys_after_migration_locked(
                    entry,
                    moved_keys,
                    force,
                )
            elif mode == "evict" and force:
                high_keys = [reservation.key for reservation in high_reservations]
                deleted_keys = self._evict_keys_locked(entry, high_keys, force=True)
                if deleted_keys:
                    self._notify_keys_deleted(
                        deleted_keys,
                        [self._config.slot_bytes] * len(deleted_keys),
                    )
            else:
                raise L2ReconfigureError(
                    409,
                    "resize shrink would drop live keys without migration",
                )
        finally:
            entry.core.finalize_reads(reservations, touched_keys)

        if not entry.core.can_shrink_to(new_max_slots):
            raise L2ReconfigureError(409, "resize shrink still has live high slots")

    def _recalculate_capacity_locked(self) -> None:
        self._max_capacity_bytes = sum(
            entry.max_dax_size_bytes
            for entry in self._devices
            if entry.state in _CAPACITY_STATES
        )

    def _ensure_hotplug_enabled(self) -> None:
        if not self._hotplug_enabled:
            raise L2ReconfigureError(403, "DAX hotplug is disabled")

    @staticmethod
    def _validate_remove_mode(mode: str) -> None:
        if mode not in {"migrate", "evict", "drain"}:
            raise L2ReconfigureError(400, "mode must be migrate, evict, or drain")

    @staticmethod
    def _validate_resize_mode(mode: str) -> None:
        if mode not in {"migrate", "evict"}:
            raise L2ReconfigureError(400, "mode must be migrate or evict")


register_l2_adapter_type("dax", DaxL2AdapterConfig)


def _create_dax_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    del l1_memory_desc
    return DaxL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("dax", _create_dax_adapter)
