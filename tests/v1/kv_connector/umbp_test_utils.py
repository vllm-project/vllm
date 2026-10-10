# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte-copy runtime for UMBP connector contract tests."""

import ctypes
import threading
from collections.abc import Sequence
from types import SimpleNamespace

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
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeFactory,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
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


def _kv_cache_config() -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8192,
                layers=["layer1", "layer2"],
                layer_stride=4096,
                block_stride=1024,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer1", "layer2"], spec)],
    )


def _hybrid_kv_cache_config() -> KVCacheConfig:
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((4,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )


def _vllm_config(
    extra: dict, prefix_match_unit: int | None = None, **parallel_overrides
) -> SimpleNamespace:
    parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "world_size": 1,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=True,
            prefix_match_unit=prefix_match_unit,
        ),
        model_config=SimpleNamespace(
            model="test-model",
            revision="r1",
            max_model_len=128,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        speculative_config=None,
        kv_events_config=None,
        max_in_flight_tokens=64,
        num_prefill_lookahead_tokens=0,
        parallel_config=SimpleNamespace(**parallel),
    )


class _SchedulerHandle:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def lookup(self, keys):
        self.queries.append(list(keys))
        if isinstance(self.hits, dict):
            return [self.hits.get(key, False) for key in keys]
        return self.hits or [False] * len(keys)

    def close(self):
        pass

    def clear(self):
        self.hits = []
        return True


class _WorkerHandle:
    def register_buffers(self, kv_caches):
        self.kv_caches = kv_caches

    def load(self, plans):
        self.loaded_plans = list(plans)
        job = TransferJobState(tuple(plans))
        job.start()
        job.complete()
        return job

    def store(self, plans):
        self.stored_plans = list(plans)
        job = TransferJobState(tuple(plans))
        job.start()
        job.complete()
        return job

    def wait(self, job):
        return job

    def poll(self, job):
        if job.status in (
            TransferJobStatus.COMPLETED,
            TransferJobStatus.FAILED,
            TransferJobStatus.CANCELLED,
        ):
            return job
        return None

    def publish(self, job):
        self.published = job

    def close(self):
        pass


class _LayerRecordingWorkerHandle(_WorkerHandle):
    def __init__(self, fail_first_load=False):
        self.load_calls = []
        self.wait_calls = []
        self.fail_first_load = fail_first_load

    def load(self, plans):
        self.load_calls.append(list(plans))
        job = super().load(plans)
        if self.fail_first_load and len(self.load_calls) == 1:
            job.completed_keys.clear()
            job.fail([plan.key for plan in plans], "first layer failed")
        return job

    def wait(self, job):
        self.wait_calls.append(job)
        return super().wait(job)

    def poll(self, job):
        return None


class _StoreRecordingWorkerHandle(_LayerRecordingWorkerHandle):
    def __init__(self):
        super().__init__()
        self.store_calls = []

    def store(self, plans):
        self.store_calls.append(list(plans))
        return super().store(plans)

    def poll(self, job):
        return _WorkerHandle.poll(self, job)


class _DelayedLoadWorkerHandle(_WorkerHandle):
    def load(self, plans):
        self.load_job = TransferJobState(tuple(plans))
        self.load_job.start()
        return self.load_job


class _WaitRecordingDelayedLoadHandle(_DelayedLoadWorkerHandle):
    def __init__(self):
        self.waited = []

    def wait(self, job):
        self.waited.append(job)
        job.complete()
        return job


class _CancellableWorkerHandle(_WorkerHandle):
    def __init__(self):
        self.cancelled = []

    def cancel(self, job):
        self.cancelled.append(job)
        job.cancel("preempted")
        return job


class _DelayedStoreWorkerHandle(_CancellableWorkerHandle):
    def __init__(self):
        super().__init__()
        self.jobs = []
        self.waited = []
        self.publications = []

    def store(self, plans):
        job = TransferJobState(tuple(plans))
        job.start()
        self.jobs.append(job)
        return job

    def wait(self, job):
        self.waited.append(job)
        job.complete()
        return job

    def publish(self, job):
        self.publications.append(job)


class _EmbeddedSchedulerHandle:
    def __init__(self, store):
        self.store = store

    def lookup(self, keys):
        return [key in self.store for key in keys]

    def close(self):
        pass


class _EmbeddedWorkerHandle(_WorkerHandle):
    def __init__(self, store):
        self.keys = store

    def load(self, plans):
        job = TransferJobState(tuple(plans))
        job.start()
        present = [plan.key for plan in plans if plan.key in self.keys]
        missing = [plan.key for plan in plans if plan.key not in self.keys]
        job.complete(present)
        if missing:
            job.fail(missing, "embedded key missing")
        return job

    def store(self, plans):
        job = TransferJobState(tuple(plans))
        job.start()
        self.keys.update(plan.key for plan in plans)
        job.complete()
        return job


class _EmbeddedRuntime:
    def __init__(self):
        self.store = set()
        self.scheduler_args = None
        self.worker_args = None

    def create_scheduler_handle(self, namespace, topology, layout):
        self.scheduler_args = (namespace, topology, layout)
        return _EmbeddedSchedulerHandle(self.store)

    def create_worker_handle(self, namespace, topology, layout):
        self.worker_args = (namespace, topology, layout)
        return _EmbeddedWorkerHandle(self.store)
