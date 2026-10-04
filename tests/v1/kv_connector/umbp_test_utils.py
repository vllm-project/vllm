# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte-copy runtime for UMBP connector contract tests."""

import ctypes
import threading
from collections.abc import Sequence

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.base import (
    IUMBPRuntime,
    UMBPRuntimeCapabilities,
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeFactory,
)


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


class _MemorySchedulerHandle(UMBPSchedulerHandle):
    def __init__(self, store: _EmbeddedStore) -> None:
        self._store = store

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        return [self._store.contains(key) for key in keys]

    def clear(self) -> bool:
        self._store.clear()
        return True

    def close(self) -> None:
        return


class _MemoryWorkerHandle(UMBPWorkerHandle):
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


class MemoryRuntime(IUMBPRuntime):
    """CPU validation runtime sharing one store across connector handles."""

    capabilities = UMBPRuntimeCapabilities(
        layerwise_load=True,
    )

    def __init__(self) -> None:
        self._store = _EmbeddedStore()

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        del namespace, topology, layout
        return _MemorySchedulerHandle(self._store)

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        del namespace
        return _MemoryWorkerHandle(self._store, topology, layout)


def install_memory_runtime(monkeypatch):
    """Share one isolated store across the connectors created by a test."""
    runtime = MemoryRuntime()
    monkeypatch.setitem(UMBPRuntimeFactory._builders, "embedded", lambda _: runtime)
    return runtime
