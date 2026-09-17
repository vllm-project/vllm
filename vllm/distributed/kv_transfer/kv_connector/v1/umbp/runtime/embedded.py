# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A small in-process embedded runtime for UMBP core validation.

This implementation intentionally uses CPU tensors and a process-local byte
store.  It validates the shared connector contract without pretending that a
Python dictionary is a GPU-capable MORI-UMBP backend.
"""

from __future__ import annotations

import ctypes
import contextlib
import hashlib
import json
import os
import socket
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from collections.abc import Sequence
from pathlib import Path

import torch

from ..data import (
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
)
from .base import (
    IUMBPRuntime,
    UMBPRuntimeCapabilities,
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from .factory import UMBPRuntimeConfig


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
            payload[item.object_offset:end] = self._read_range(
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
                payload[item.object_offset:end],
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

    def publish(self, job: TransferJobState) -> None:
        if job.status.value != "completed":
            raise RuntimeError("cannot publish an incomplete embedded job")
        self._store.publish(id(job))

    def cancel(self, job: TransferJobState) -> TransferJobState:
        if job.status.value not in ("completed", "failed"):
            job.cancel("preempted")
        return job

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
    def from_config(cls, config: UMBPRuntimeConfig) -> "_MemoryEmbeddedRuntime":
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
    digest = hashlib.sha256(
        f"{namespace}:{rank_namespace}".encode()
    ).hexdigest()[:24]
    return str(Path(lookup_dir) / f"vllm-umbp-{digest}.sock")


class _MoriLookupServer:
    """Small worker-local lookup bridge for the scheduler process."""

    def __init__(self, path: str, client: object) -> None:
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
        self._paths = [
            _lookup_socket_path(namespace, rank, lookup_dir)
            for rank in topology.all_namespaces()
        ]

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        result = [False] * len(keys)
        for path in self._paths:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    sock.connect(path)
                    sock.sendall((json.dumps(list(keys)) + "\n").encode())
                    response = b""
                    while not response.endswith(b"\n"):
                        chunk = sock.recv(65536)
                        if not chunk:
                            break
                        response += chunk
                    values = json.loads(response.decode() or "[]")
                    result = [
                        current or bool(values[index])
                        for index, current in enumerate(result)
                    ]
            except (OSError, ValueError, IndexError):
                continue
        return result

    def close(self) -> None:
        return

    def clear(self) -> bool:
        success = True
        contacted = False
        for path in self._paths:
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
        client: object,
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
        pointers = [
            [item.base_address for item in plan.ranges] for plan in plans
        ]
        sizes = [[item.length for item in plan.ranges] for plan in plans]
        offsets = [
            [item.object_offset for item in plan.ranges] for plan in plans
        ]
        return keys, object_sizes, pointers, sizes, offsets

    def load(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        plans = tuple(plans)
        job = TransferJobState(plans)
        job.start()
        self._futures[id(job)] = self._executor.submit(
            self._load_sync, job, plans
        )
        return job

    def _load_sync(
        self, job: TransferJobState, plans: tuple[BlockTransferPlan, ...]
    ) -> TransferJobState:
        if not plans:
            job.complete()
            return job
        keys, _, pointers, sizes, offsets = self._range_args(plans)
        results = self.client.batch_get_ranges_into_ptr(
            keys, pointers, sizes, offsets
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
        ready_events: list[torch.cuda.Event] = []
        for device in self._gpu_devices:
            with torch.cuda.device(device):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                ready_events.append(event)
        self._futures[id(job)] = self._executor.submit(
            self._store_sync, job, plans, ready_events
        )
        return job

    def _store_sync(
        self,
        job: TransferJobState,
        plans: tuple[BlockTransferPlan, ...],
        ready_events: list[torch.cuda.Event],
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
        client_config.dram.capacity_bytes = config.options.get(
            "capacity_bytes", 64 * 1024**3
        )
        return _MoriEmbeddedRuntime(
            UMBPClient(client_config),
            config.options.get("lookup_dir", "/tmp"),
            int(config.options.get("num_workers", 4)),
            float(config.options.get("timeout_ms", 30000)) / 1000,
        )


class _MoriEmbeddedRuntime(IUMBPRuntime):
    capabilities = UMBPRuntimeCapabilities(
        ranged_io=True,
        layerwise_load=True,
        cancellation=True,
        async_transfer=True,
    )

    def __init__(
        self,
        client: object,
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
