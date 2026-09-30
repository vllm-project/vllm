# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A small in-process embedded runtime for UMBP core validation.

This implementation intentionally uses CPU tensors and a process-local byte
store.  It validates the shared connector contract without pretending that a
Python dictionary is a GPU-capable MORI-UMBP backend.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import socket
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Protocol

import regex as re
import torch

from vllm.logger import init_logger

from ..data import (
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
)
from .base import (
    IUMBPRuntime,
    UMBPRuntimeCapabilities,
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from .factory import UMBPRuntimeConfig

logger = init_logger(__name__)

_RANK_KEY_PATTERN = re.compile(r":tp(\d+):pcp(\d+):dcp(\d+):pp(\d+):g\d+:")


def _rank_namespace_from_key(key: str) -> tuple[int, int, int, int] | None:
    match = _RANK_KEY_PATTERN.search(key)
    if match is None:
        return None
    tp_rank, pcp_rank, dcp_rank, pp_rank = map(int, match.groups())
    return tp_rank, pcp_rank, dcp_rank, pp_rank


class _LookupClient(Protocol):
    def batch_exists(self, keys: Sequence[str]) -> Sequence[bool]: ...

    def clear(self) -> bool: ...


class _MoriClient(_LookupClient, Protocol):
    def register_memory(self, *args: Any) -> bool: ...

    def deregister_memory(self, *args: Any) -> bool: ...

    def batch_get_ranges_into_ptr(self, *args: Any) -> Sequence[bool]: ...

    def batch_put_ranges_from_ptr(self, *args: Any) -> Sequence[bool]: ...

    def flush(self) -> bool: ...


class _EmbeddedStore:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._staged: dict[int, dict[str, bytes]] = {}
        self._lock = threading.RLock()

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._objects

    def stage(self, job_id: int, key: str, value: bytes) -> None:
        with self._lock:
            self._staged.setdefault(job_id, {})[key] = value

    def publish(self, job_id: int) -> None:
        with self._lock:
            staged = self._staged.pop(job_id, {})
            self._objects.update(staged)

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._objects.get(key)

    def clear(self) -> None:
        with self._lock:
            self._objects.clear()
            self._staged.clear()


class EmbeddedSchedulerHandle(UMBPSchedulerHandle):
    def __init__(self, store: _EmbeddedStore) -> None:
        self._store = store

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        return [self._store.contains(key) for key in keys]

    def clear(self) -> bool:
        self._store.clear()
        return True

    def close(self) -> None:
        return


class EmbeddedWorkerHandle(UMBPWorkerHandle):
    def __init__(
        self,
        store: _EmbeddedStore,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> None:
        self._store = store
        self.topology = topology
        self.layout = layout
        self._registered = False
        self.last_load_plans: list[BlockTransferPlan] = []
        self.last_store_plans: list[BlockTransferPlan] = []

    def register_buffers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if any(cache.device.type != "cpu" for cache in kv_caches.values()):
            raise RuntimeError(
                "The validation embedded runtime supports CPU tensors only"
            )
        self._registered = True

    @staticmethod
    def _object_size(plan: BlockTransferPlan) -> int:
        return max(
            (item.object_offset + item.length for item in plan.ranges),
            default=0,
        )

    @staticmethod
    def _read_range(base_address: int, length: int) -> bytes:
        return ctypes.string_at(base_address, length)

    @staticmethod
    def _write_range(base_address: int, payload: bytes) -> None:
        ctypes.memmove(base_address, payload, len(payload))

    def _read_object(self, plan: BlockTransferPlan) -> bytes:
        if not self._registered:
            raise RuntimeError("register_buffers must be called before store")
        payload = bytearray(self._object_size(plan))
        for item in plan.ranges:
            end = item.object_offset + item.length
            payload[item.object_offset : end] = self._read_range(
                item.base_address, item.length
            )
        return bytes(payload)

    def _write_object(self, plan: BlockTransferPlan, payload: bytes) -> None:
        if not self._registered:
            raise RuntimeError("register_buffers must be called before load")
        for item in plan.ranges:
            end = item.object_offset + item.length
            if end > len(payload):
                raise ValueError(f"stored object is too small for key {plan.key}")
            self._write_range(
                item.base_address,
                payload[item.object_offset : end],
            )

    def load(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        self.last_load_plans = list(plans)
        job = TransferJobState(tuple(plans))
        job.start()
        completed: list[str] = []
        failed: list[str] = []
        for plan in plans:
            payload = self._store.get(plan.key)
            if payload is None:
                failed.append(plan.key)
                continue
            try:
                self._write_object(plan, payload)
                completed.append(plan.key)
            except Exception:
                failed.append(plan.key)
        if completed:
            job.complete(completed)
        if failed:
            job.fail(failed, "embedded object load failed")
        return job

    def store(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        self.last_store_plans = list(plans)
        job = TransferJobState(tuple(plans))
        job.start()
        completed: list[str] = []
        failed: list[str] = []
        for plan in plans:
            try:
                self._store.stage(id(job), plan.key, self._read_object(plan))
                completed.append(plan.key)
            except Exception:
                failed.append(plan.key)
        if completed:
            job.complete(completed)
        if failed:
            job.fail(failed, "embedded object store failed")
        return job

    def wait(self, job: TransferJobState) -> TransferJobState:
        return job

    def poll(self, job: TransferJobState) -> TransferJobState | None:
        if job.status in (
            TransferJobStatus.COMPLETED,
            TransferJobStatus.FAILED,
            TransferJobStatus.CANCELLED,
        ):
            return job
        return None

    def publish(self, job: TransferJobState) -> None:
        if job.status.value != "completed":
            raise RuntimeError("cannot publish an incomplete embedded job")
        self._store.publish(id(job))

    def cancel(self, job: TransferJobState) -> TransferJobState:
        if job.status.value not in ("completed", "failed"):
            job.cancel("preempted")
        return job

    def take_evicted_keys(self) -> Sequence[str]:
        return ()

    def close(self) -> None:
        return


class _MemoryEmbeddedRuntime(IUMBPRuntime):
    """CPU validation runtime sharing one store across connector handles."""

    capabilities = UMBPRuntimeCapabilities(
        ranged_io=True,
        layerwise_load=True,
        cancellation=True,
    )
    _store = _EmbeddedStore()

    @classmethod
    def from_config(cls, config: UMBPRuntimeConfig) -> _MemoryEmbeddedRuntime:
        del config
        return cls()

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        del namespace, topology, layout
        return EmbeddedSchedulerHandle(self._store)

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        del namespace
        return EmbeddedWorkerHandle(self._store, topology, layout)


def _lookup_socket_path(
    namespace: str,
    rank_namespace: tuple[int, int, int, int],
    lookup_dir: str,
) -> str:
    digest = hashlib.sha256(f"{namespace}:{rank_namespace}".encode()).hexdigest()[:24]
    return str(Path(lookup_dir) / f"vllm-umbp-{digest}.sock")


class _MoriLookupServer:
    """Small worker-local lookup bridge for the scheduler process."""

    def __init__(self, path: str, client: _LookupClient) -> None:
        self.path = path
        self.client = client
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(self.path)
        self._socket.listen(16)
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._serve,
            name="umbp-embedded-lookup",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                try:
                    request = b""
                    while not request.endswith(b"\n"):
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        request += chunk
                    payload = json.loads(request.decode())
                    if isinstance(payload, dict) and payload.get("op") == "clear":
                        connection.sendall(
                            (json.dumps(bool(self.client.clear())) + "\n").encode()
                        )
                        continue
                    result = self.client.batch_exists(payload)
                    connection.sendall(
                        (json.dumps([bool(value) for value in result]) + "\n").encode()
                    )
                except Exception:
                    with contextlib.suppress(OSError):
                        connection.sendall(b"[]\n")

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)


class _MoriSchedulerHandle(UMBPSchedulerHandle):
    def __init__(
        self,
        namespace: str,
        topology: RankTopology,
        lookup_dir: str,
    ) -> None:
        self._paths = {
            rank: _lookup_socket_path(namespace, rank, lookup_dir)
            for rank in topology.all_namespaces()
        }
        self.last_lookup_diagnostics: dict[str, tuple[Any, ...]] = {}

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        result = [False] * len(keys)
        keys_by_rank: dict[tuple[int, int, int, int], list[tuple[int, str]]] = {}
        unrouted: list[tuple[int, str]] = []
        for index, key in enumerate(keys):
            rank = _rank_namespace_from_key(key)
            if rank is None or rank not in self._paths:
                unrouted.append((index, key))
            else:
                keys_by_rank.setdefault(rank, []).append((index, key))
        unavailable: list[tuple[int, int, int, int]] = []
        for rank, indexed_keys in keys_by_rank.items():
            path = self._paths[rank]
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    sock.connect(path)
                    sock.sendall(
                        (json.dumps([key for _, key in indexed_keys]) + "\n").encode()
                    )
                    response = b""
                    while not response.endswith(b"\n"):
                        chunk = sock.recv(65536)
                        if not chunk:
                            break
                        response += chunk
                    values = json.loads(response.decode() or "[]")
                    if len(values) != len(indexed_keys):
                        raise ValueError("lookup returned an invalid result length")
                    for (index, _), value in zip(indexed_keys, values, strict=True):
                        result[index] = bool(value)
            except (OSError, ValueError, IndexError):
                unavailable.append(rank)
        for path in self._paths.values():
            if not unrouted:
                break
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    sock.connect(path)
                    sock.sendall(
                        (json.dumps([key for _, key in unrouted]) + "\n").encode()
                    )
                    response = b""
                    while not response.endswith(b"\n"):
                        chunk = sock.recv(65536)
                        if not chunk:
                            break
                        response += chunk
                    values = json.loads(response.decode() or "[]")
                    for (index, _), value in zip(unrouted, values, strict=True):
                        result[index] = result[index] or bool(value)
            except (OSError, ValueError):
                continue
        missing = tuple(
            key for key, exists in zip(keys, result, strict=True) if not exists
        )
        self.last_lookup_diagnostics = {
            "unavailable_ranks": tuple(unavailable),
            "missing_keys": missing,
        }
        if unavailable:
            logger.warning("UMBP lookup unavailable for ranks: %s", unavailable)
        elif missing:
            logger.debug("UMBP lookup missing %d objects", len(missing))
        return result

    def close(self) -> None:
        return

    def clear(self) -> bool:
        success = True
        contacted = False
        for path in self._paths.values():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(2.0)
                    sock.connect(path)
                    sock.sendall(b'{"op":"clear"}\n')
                    response = b""
                    while not response.endswith(b"\n"):
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response += chunk
                    contacted = True
                    success = success and bool(json.loads(response.decode()))
            except (OSError, ValueError):
                success = False
        return contacted and success


class _MoriWorkerHandle(UMBPWorkerHandle):
    def __init__(
        self,
        client: _MoriClient,
        namespace: str,
        topology: RankTopology,
        lookup_dir: str,
        max_workers: int,
        timeout_s: float,
    ) -> None:
        self.client = client
        self._namespace = namespace
        self._topology = topology
        self._lookup_dir = lookup_dir
        self._lookup_server: _MoriLookupServer | None = None
        self._registered_storages: set[int] = set()
        self._gpu_devices: set[int] = set()
        self._published_keys: set[str] = set()
        self._evicted_keys: set[str] = set()
        self._key_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="umbp-embedded-transfer",
        )
        self._futures: dict[int, Future[TransferJobState]] = {}
        self._timeout_s = timeout_s
        self._store_retry_count = 2
        self._store_retry_backoff_s = 0.01

    def register_buffers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            from mori.cpp import MemoryLocationType
        except ImportError as exc:
            raise RuntimeError("MORI UMBP Python bindings are unavailable") from exc

        for cache in kv_caches.values():
            storage = cache.untyped_storage()
            storage_ptr = storage.data_ptr()
            if storage_ptr in self._registered_storages:
                continue
            location = (
                MemoryLocationType.GPU
                if cache.device.type == "cuda"
                else MemoryLocationType.CPU
            )
            device = cache.device.index if cache.device.index is not None else -1
            if not self.client.register_memory(
                storage_ptr, storage.nbytes(), location, device
            ):
                raise RuntimeError(
                    f"MORI UMBP failed to register KV storage 0x{storage_ptr:x}"
                )
            self._registered_storages.add(storage_ptr)
            if location == MemoryLocationType.GPU and device >= 0:
                self._gpu_devices.add(device)

        if self._lookup_server is None:
            self._lookup_server = _MoriLookupServer(
                _lookup_socket_path(
                    self._namespace,
                    self._topology.local_namespace,
                    self._lookup_dir,
                ),
                self,
            )
            self._lookup_server.start()

    def batch_exists(self, keys: Sequence[str]) -> Sequence[bool]:
        result = [bool(value) for value in self.client.batch_exists(keys)]
        with self._key_lock:
            for key, exists in zip(keys, result, strict=True):
                if not exists and key in self._published_keys:
                    self._published_keys.remove(key)
                    self._evicted_keys.add(key)
        return result

    def clear(self) -> bool:
        success = bool(self.client.clear())
        if success:
            with self._key_lock:
                self._published_keys.clear()
                self._evicted_keys.clear()
        return success

    @staticmethod
    def _range_args(plans: Sequence[BlockTransferPlan]):
        keys = [plan.key for plan in plans]
        object_sizes = [
            max(
                (item.object_offset + item.length for item in plan.ranges),
                default=0,
            )
            for plan in plans
        ]
        pointers = [[item.base_address for item in plan.ranges] for plan in plans]
        sizes = [[item.length for item in plan.ranges] for plan in plans]
        offsets = [[item.object_offset for item in plan.ranges] for plan in plans]
        return keys, object_sizes, pointers, sizes, offsets

    def load(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        plans = tuple(plans)
        job = TransferJobState(plans)
        job.start()
        self._futures[id(job)] = self._executor.submit(self._load_sync, job, plans)
        return job

    def _load_sync(
        self, job: TransferJobState, plans: tuple[BlockTransferPlan, ...]
    ) -> TransferJobState:
        if not plans:
            job.complete()
            return job
        keys, _, pointers, sizes, offsets = self._range_args(plans)
        started_at = time.monotonic()
        logger.debug(
            "MORI UMBP range load started request=%s plans=%d ranges=%d bytes=%d",
            plans[0].request_id,
            len(plans),
            sum(len(plan_sizes) for plan_sizes in sizes),
            sum(sum(plan_sizes) for plan_sizes in sizes),
        )
        results = self.client.batch_get_ranges_into_ptr(keys, pointers, sizes, offsets)
        logger.debug(
            "MORI UMBP range load returned request=%s plans=%d elapsed=%.3fs",
            plans[0].request_id,
            len(plans),
            time.monotonic() - started_at,
        )
        completed = [plan.key for plan, ok in zip(plans, results, strict=True) if ok]
        failed = [plan.key for plan, ok in zip(plans, results, strict=True) if not ok]
        if completed:
            job.complete(completed)
        if failed:
            job.fail(failed, "MORI UMBP range load failed")
        return job

    def store(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        plans = tuple(plans)
        job = TransferJobState(plans)
        job.start()
        ready_events: list[torch.Event] = []
        for device in self._gpu_devices:
            with torch.device(f"cuda:{device}"):
                event = torch.Event()
                event.record(torch.accelerator.current_stream())
                ready_events.append(event)
        self._futures[id(job)] = self._executor.submit(
            self._store_sync, job, plans, ready_events
        )
        return job

    def _store_sync(
        self,
        job: TransferJobState,
        plans: tuple[BlockTransferPlan, ...],
        ready_events: list[torch.Event],
    ) -> TransferJobState:
        for event in ready_events:
            event.synchronize()
        if not plans:
            job.complete()
            return job
        keys, object_sizes, pointers, sizes, offsets = self._range_args(plans)
        results = self.client.batch_put_ranges_from_ptr(
            keys, object_sizes, pointers, sizes, offsets
        )
        failed_indices = [index for index, ok in enumerate(results) if not ok]
        for attempt in range(self._store_retry_count):
            if not failed_indices:
                break
            self.client.flush()
            if self._store_retry_backoff_s:
                time.sleep(self._store_retry_backoff_s * (attempt + 1))
            for index in failed_indices.copy():
                retry = self.client.batch_put_ranges_from_ptr(
                    [keys[index]],
                    [object_sizes[index]],
                    [pointers[index]],
                    [sizes[index]],
                    [offsets[index]],
                )
                if retry and retry[0]:
                    results[index] = True
                    failed_indices.remove(index)
            if failed_indices:
                logger.warning(
                    "MORI UMBP store retry %d left %d/%d objects pending",
                    attempt + 1,
                    len(failed_indices),
                    len(plans),
                )
        completed = [plan.key for plan, ok in zip(plans, results, strict=True) if ok]
        failed = [plan.key for plan, ok in zip(plans, results, strict=True) if not ok]
        if completed:
            job.complete(completed)
        if failed:
            job.fail(failed, "MORI UMBP range store failed")
        return job

    def wait(self, job: TransferJobState) -> TransferJobState:
        future = self._futures.pop(id(job), None)
        if future is None:
            return job
        try:
            return future.result(timeout=self._timeout_s)
        except TimeoutError:
            job.fail(
                [plan.key for plan in job.plans],
                "MORI UMBP transfer timed out",
            )
            return job
        except Exception as exc:
            job.fail([plan.key for plan in job.plans], str(exc))
            return job

    def poll(self, job: TransferJobState) -> TransferJobState | None:
        future = self._futures.get(id(job))
        if future is None:
            if job.status in (
                TransferJobStatus.COMPLETED,
                TransferJobStatus.FAILED,
                TransferJobStatus.CANCELLED,
            ):
                return job
            return None
        if not future.done():
            return None
        self._futures.pop(id(job), None)
        try:
            return future.result()
        except Exception as exc:
            job.fail([plan.key for plan in job.plans], str(exc))
            return job

    def publish(self, job: TransferJobState) -> None:
        if job.status.value != "completed":
            raise RuntimeError("cannot publish an incomplete MORI UMBP job")
        if not self.client.flush():
            raise RuntimeError("MORI UMBP flush failed")
        with self._key_lock:
            self._published_keys.update(job.completed_keys)

    def cancel(self, job: TransferJobState) -> TransferJobState:
        future = self._futures.pop(id(job), None)
        if future is None:
            return job
        if future.cancel():
            job.cancel("preempted")
            return job
        try:
            return future.result(timeout=self._timeout_s)
        except TimeoutError:
            job.fail(
                [plan.key for plan in job.plans],
                "MORI UMBP cancellation timed out",
            )
            return job
        except Exception as exc:
            job.fail([plan.key for plan in job.plans], str(exc))
            return job

    def take_evicted_keys(self) -> Sequence[str]:
        with self._key_lock:
            result = tuple(self._evicted_keys)
            self._evicted_keys.clear()
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._lookup_server is not None:
            self._lookup_server.close()
        for storage_ptr in self._registered_storages:
            self.client.deregister_memory(storage_ptr)
        self._registered_storages.clear()
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        else:
            self.client.flush()


class EmbeddedRuntime(IUMBPRuntime):
    """MORI-backed embedded runtime, with explicit memory test fallback."""

    capabilities = UMBPRuntimeCapabilities(
        ranged_io=True,
        layerwise_load=True,
        partial_hash_hits=True,
        cancellation=True,
        async_transfer=True,
    )

    @classmethod
    def from_config(cls, config: UMBPRuntimeConfig) -> IUMBPRuntime:
        if config.options.get("backend") == "memory":
            return _MemoryEmbeddedRuntime.from_config(config)
        try:
            from mori.cpp import UMBPClient, UMBPConfig
        except ImportError as exc:
            raise RuntimeError(
                "Embedded UMBP requires MORI built with BUILD_UMBP=ON"
            ) from exc

        client_config = UMBPConfig()
        _configure_dram(client_config, config.options)
        total = config.options.get("_configured_total_capacity_bytes")
        if total is None:
            logger.info(
                "Embedded UMBP DRAM capacity: %.2f GiB per rank",
                client_config.dram.capacity_bytes / 1024**3,
            )
        else:
            logger.info(
                "Embedded UMBP DRAM capacity: %.2f GiB total, %.2f GiB per rank",
                total / 1024**3,
                client_config.dram.capacity_bytes / 1024**3,
            )
        return _MoriEmbeddedRuntime(
            UMBPClient(client_config),
            config.options.get("lookup_dir", "/tmp"),
            int(config.options.get("num_workers", 4)),
            float(config.options.get("timeout_ms", 30000)) / 1000,
        )


def _configure_dram(client_config: Any, options: dict[str, Any]) -> None:
    """Apply embedded-only host-DRAM options to MORI's client config."""
    dram = client_config.dram
    dram.capacity_bytes = options.get("capacity_bytes", 64 * 1024**3)
    option_fields = {
        "dram_use_shared_memory": "use_shared_memory",
        "dram_shm_name": "shm_name",
        "dram_high_watermark": "high_watermark",
        "dram_low_watermark": "low_watermark",
        "dram_use_hugepages": "use_hugepages",
        "dram_hugepage_size": "hugepage_size",
        "dram_numa_node": "numa_node",
        "dram_prefault": "prefault",
    }
    for option, field in option_fields.items():
        if option in options:
            setattr(dram, field, options[option])


class _MoriEmbeddedRuntime(IUMBPRuntime):
    capabilities = UMBPRuntimeCapabilities(
        ranged_io=True,
        layerwise_load=True,
        partial_hash_hits=True,
        cancellation=True,
        async_transfer=True,
    )

    def __init__(
        self,
        client: _MoriClient,
        lookup_dir: str,
        max_workers: int,
        timeout_s: float,
    ) -> None:
        self.client = client
        self.lookup_dir = lookup_dir
        self.max_workers = max_workers
        self.timeout_s = timeout_s

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        del layout
        return _MoriSchedulerHandle(namespace, topology, self.lookup_dir)

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        del layout
        return _MoriWorkerHandle(
            self.client,
            namespace,
            topology,
            self.lookup_dir,
            self.max_workers,
            self.timeout_s,
        )
