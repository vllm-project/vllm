# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MORI-backed embedded DRAM offloading."""

from __future__ import annotations

import contextlib
import hashlib
import os
import socket
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import msgspec
import numpy as np
import regex as re
import torch

from vllm.logger import init_logger

from ..data import (
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
)
from .base import IUMBPRuntime, UMBPSchedulerHandle, UMBPWorkerHandle
from .factory import UMBPRuntimeConfig

logger = init_logger(__name__)

# Bounds one scheduler-to-worker lookup round trip on the local socket.
_LOOKUP_TIMEOUT_S = 1.0

# MORI evicts asynchronously once the pool crosses its high watermark, so a
# put into a full pool can fail until eviction catches up.
_STORE_RETRIES = 2
_STORE_RETRY_BACKOFF_S = 0.01

_RANK_KEY_PATTERN = re.compile(r":tp(\d+):pcp(\d+):dcp(\d+):pp(\d+):g\d+:")


@lru_cache(maxsize=256)
def _rank_namespace_from_key_prefix(prefix: str) -> tuple[int, int, int, int] | None:
    match = _RANK_KEY_PATTERN.search(prefix)
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

    def batch_put_ranges_from_ptr(self, *args: Any) -> list[bool]: ...

    def flush(self) -> bool: ...


def _lookup_socket_path(
    namespace: str,
    rank_namespace: tuple[int, int, int, int],
    lookup_dir: str,
    instance: str = "",
) -> str:
    # The namespace identifies the model and layout, not the engine: DP ranks
    # and independent engines serving one model share it, so the engine's own
    # identity must separate their private pools.
    identity = f"{namespace}:{rank_namespace}"
    if instance:
        identity = f"{namespace}:{instance}:{rank_namespace}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return str(Path(lookup_dir) / f"vllm-umbp-{digest}.sock")


def _socket_has_listener(path: str) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(path)
    except OSError:
        return False
    finally:
        probe.close()
    return True


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
        if os.path.exists(self.path) and _socket_has_listener(self.path):
            raise RuntimeError(
                f"UMBP lookup socket {self.path} is served by another live "
                "engine; set lookup_instance to a distinct value for each "
                "engine that serves this model on this host"
            )
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
                connection.settimeout(_LOOKUP_TIMEOUT_S)
                try:
                    request = b""
                    while not request.endswith(b"\n"):
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        request += chunk
                    payload = msgspec.json.decode(request)
                    if isinstance(payload, dict) and payload.get("op") == "clear":
                        connection.sendall(
                            msgspec.json.encode(bool(self.client.clear())) + b"\n"
                        )
                        continue
                    result = self.client.batch_exists(payload)
                    connection.sendall(
                        msgspec.json.encode([bool(value) for value in result]) + b"\n"
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
        instance: str = "",
    ) -> None:
        self._paths = {
            rank: _lookup_socket_path(namespace, rank, lookup_dir, instance)
            for rank in topology.all_namespaces()
        }
        self.last_lookup_diagnostics: dict[str, tuple[Any, ...]] = {}

    @staticmethod
    def _request(path: str, payload: Any, timeout: float = _LOOKUP_TIMEOUT_S) -> Any:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(path)
            sock.sendall(msgspec.json.encode(payload) + b"\n")
            response = b""
            while not response.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                response += chunk
            return msgspec.json.decode(response or b"[]")

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        unavailable: list[tuple[int, int, int, int]]
        if len(self._paths) == 1:
            rank, path = next(iter(self._paths.items()))
            try:
                values = self._request(path, keys)
                if len(values) != len(keys):
                    raise ValueError("lookup returned an invalid result length")
                return self._lookup_result(keys, [bool(value) for value in values], [])
            except (OSError, ValueError, IndexError):
                unavailable = (
                    [rank]
                    if any(
                        _rank_namespace_from_key_prefix(key.rpartition(":")[0] + ":")
                        == rank
                        for key in keys
                    )
                    else []
                )
                return self._lookup_result(keys, [False] * len(keys), unavailable)
        result = [False] * len(keys)
        keys_by_rank: dict[tuple[int, int, int, int], list[tuple[int, str]]] = {}
        unrouted: list[tuple[int, str]] = []
        for index, key in enumerate(keys):
            key_rank = _rank_namespace_from_key_prefix(key.rpartition(":")[0] + ":")
            if key_rank is None or key_rank not in self._paths:
                unrouted.append((index, key))
            else:
                keys_by_rank.setdefault(key_rank, []).append((index, key))
        unavailable = []
        for rank, indexed_keys in keys_by_rank.items():
            path = self._paths[rank]
            try:
                values = self._request(path, [key for _, key in indexed_keys])
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
                values = self._request(path, [key for _, key in unrouted])
                for (index, _), value in zip(unrouted, values, strict=True):
                    result[index] = result[index] or bool(value)
            except (OSError, ValueError):
                continue
        return self._lookup_result(keys, result, unavailable)

    def _lookup_result(
        self,
        keys: Sequence[str],
        result: list[bool],
        unavailable: list[tuple[int, int, int, int]],
    ) -> list[bool]:
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
                cleared = self._request(path, {"op": "clear"}, timeout=2.0)
                contacted = True
                success = success and bool(cleared)
            except (OSError, ValueError):
                success = False
        return contacted and success


class _BlockLoadLayout(NamedTuple):
    bases: np.ndarray
    strides: np.ndarray
    sizes: list[int]
    offsets: list[int]
    num_blocks: int
    num_bytes: int


_LoadArgs = tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]
_StoreArgs = tuple[
    list[str], list[int], list[list[int]], list[list[int]], list[list[int]]
]


class _MoriWorkerHandle(UMBPWorkerHandle):
    def __init__(
        self,
        client: _MoriClient,
        namespace: str,
        topology: RankTopology,
        lookup_dir: str,
        max_workers: int,
        timeout_s: float,
        layout: KVLayoutDescriptor | None = None,
        lookup_instance: str = "",
        serve_lookups: bool = True,
    ) -> None:
        self.client = client
        self._namespace = namespace
        self._topology = topology
        self._lookup_dir = lookup_dir
        self._lookup_instance = lookup_instance
        self._serve_lookups = serve_lookups
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
        self._store_deadlines: dict[int, float] = {}
        self._timeout_s = timeout_s
        self._layout = layout
        self._load_layouts: dict[int, _BlockLoadLayout] = {}

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

        self._register_load_layouts(kv_caches)
        if self._serve_lookups and self._lookup_server is None:
            self._lookup_server = _MoriLookupServer(
                _lookup_socket_path(
                    self._namespace,
                    self._topology.local_namespace,
                    self._lookup_dir,
                    self._lookup_instance,
                ),
                self,
            )
            self._lookup_server.start()

    def _register_load_layouts(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._load_layouts = {}
        if self._layout is None:
            return
        for group_id in {r.group_id for r in self._layout.regions}:
            bases: list[int] = []
            strides: list[int] = []
            sizes: list[int] = []
            offsets: list[int] = []
            counts: list[int] = []
            for region in self._layout.regions:
                if region.group_id != group_id:
                    continue
                cache = kv_caches.get(region.layer_name)
                if cache is None or cache.ndim == 0:
                    break
                stride = cache.stride(0) * cache.element_size()
                if (
                    stride <= 0
                    or stride > region.block_stride
                    or region.block_stride % stride
                    or region.block_bytes > stride
                ):
                    break
                bases.append(cache.data_ptr())
                strides.append(stride)
                offsets.append(sum(sizes))
                sizes.append(region.block_bytes)
                counts.append(cache.shape[0])
            else:
                self._load_layouts[group_id] = _BlockLoadLayout(
                    np.asarray(bases, dtype=np.uint64),
                    np.asarray(strides, dtype=np.uint64),
                    sizes,
                    offsets,
                    min(counts),
                    sum(sizes),
                )

    def _bulk_load_args(
        self, plans: Sequence[BlockTransferPlan]
    ) -> tuple[_LoadArgs, tuple[int, ...]] | None:
        groups: dict[int, list[tuple[int, int]]] = {}
        entries: Iterable[tuple[int | None, int]]
        if isinstance(plans, BlockLoadBatch):
            entries = zip(plans.group_ids, plans.block_ids, strict=True)
            keys = plans.keys
        else:
            if any(p.ranges for p in plans):
                return None
            entries = ((p.group_id, p.block_id) for p in plans)
            keys = [p.key for p in plans]
        for index, (group_id, block_id) in enumerate(entries):
            if group_id is None:
                return None
            layout = self._load_layouts.get(group_id)
            if layout is None or not 0 <= block_id < layout.num_blocks:
                return None
            groups.setdefault(group_id, []).append((index, block_id))
        pointers: list[list[int]] = [[]] * len(plans)
        sizes: list[list[int]] = [[]] * len(plans)
        offsets: list[list[int]] = [[]] * len(plans)
        plan_bytes = [0] * len(plans)
        for group_id, group_entries in groups.items():
            layout = self._load_layouts[group_id]
            ids = np.asarray([block for _, block in group_entries], dtype=np.uint64)
            addresses = (
                layout.bases[None, :] + ids[:, None] * layout.strides[None, :]
            ).tolist()
            for (index, _), row in zip(group_entries, addresses, strict=True):
                pointers[index] = row
                sizes[index] = layout.sizes
                offsets[index] = layout.offsets
                plan_bytes[index] = layout.num_bytes
        return (
            (keys, pointers, sizes, offsets),
            tuple(plan_bytes),
        )

    def load_blocks(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        prepared = self._bulk_load_args(plans)
        if prepared is None:
            return None
        args, plan_bytes = prepared
        return self._submit_load(
            plans if isinstance(plans, BlockLoadBatch) else tuple(plans),
            args,
            plan_bytes,
        )

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
    def _range_args(plans: Sequence[BlockTransferPlan], *, for_store: bool = True):
        keys, object_sizes, pointers, sizes, offsets = [], [], [], [], []
        for plan in plans:
            plan_pointers, plan_sizes, plan_offsets = [], [], []
            object_size = 0
            for item in plan.ranges:
                plan_pointers.append(item.base_address)
                plan_sizes.append(item.length)
                plan_offsets.append(item.object_offset)
                if for_store:
                    end = item.object_offset + item.length
                    if end > object_size:
                        object_size = end
            keys.append(plan.key)
            if for_store:
                object_sizes.append(object_size)
            pointers.append(plan_pointers)
            sizes.append(plan_sizes)
            offsets.append(plan_offsets)
        return keys, object_sizes, pointers, sizes, offsets

    def load(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        return self._submit_load(tuple(plans))

    def _submit_load(
        self,
        plans: tuple[BlockTransferPlan, ...] | BlockLoadBatch,
        range_args: _LoadArgs | None = None,
        plan_bytes: tuple[int, ...] | None = None,
    ) -> TransferJobState:
        job = TransferJobState(plans, plan_bytes=plan_bytes)
        job.start()
        self._futures[id(job)] = self._executor.submit(
            self._load_sync, job, plans, range_args
        )
        return job

    def _load_sync(
        self,
        job: TransferJobState,
        plans: tuple[BlockTransferPlan, ...] | BlockLoadBatch,
        range_args: _LoadArgs | None = None,
    ) -> TransferJobState:
        if not plans:
            job.complete()
            return job
        if range_args is None:
            keys, _, pointers, sizes, offsets = self._range_args(plans, for_store=False)
        else:
            keys, pointers, sizes, offsets = range_args
        request_id = (
            plans.request_id
            if isinstance(plans, BlockLoadBatch)
            else plans[0].request_id
        )
        started_at = time.monotonic()
        logger.debug(
            "MORI UMBP range load started request=%s plans=%d ranges=%d bytes=%d",
            request_id,
            len(plans),
            sum(len(plan_sizes) for plan_sizes in sizes),
            sum(sum(plan_sizes) for plan_sizes in sizes),
        )
        results = self.client.batch_get_ranges_into_ptr(keys, pointers, sizes, offsets)
        logger.debug(
            "MORI UMBP range load returned request=%s plans=%d elapsed=%.3fs",
            request_id,
            len(plans),
            time.monotonic() - started_at,
        )
        completed = [key for key, ok in zip(keys, results, strict=True) if ok]
        failed = [key for key, ok in zip(keys, results, strict=True) if not ok]
        if completed:
            job.complete(completed)
        if failed:
            job.fail(failed, "MORI UMBP range load failed")
        return job

    def store_blocks(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        """Store whole blocks from registered layouts; None requests ranges."""
        prepared = self._bulk_load_args(plans)
        if prepared is None:
            return None
        (keys, pointers, sizes, offsets), plan_bytes = prepared
        return self._submit_store(
            tuple(plans),
            (keys, list(plan_bytes), pointers, sizes, offsets),
            plan_bytes,
        )

    def store(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        return self._submit_store(tuple(plans))

    def _submit_store(
        self,
        plans: tuple[BlockTransferPlan, ...],
        range_args: _StoreArgs | None = None,
        plan_bytes: tuple[int, ...] | None = None,
    ) -> TransferJobState:
        job = TransferJobState(plans, plan_bytes=plan_bytes)
        job.start()
        ready_events: list[torch.Event] = []
        for device in self._gpu_devices:
            with torch.device(f"cuda:{device}"):
                event = torch.Event()
                event.record(torch.accelerator.current_stream())
                ready_events.append(event)
        self._futures[id(job)] = self._executor.submit(
            self._store_sync, job, plans, ready_events, range_args
        )
        return job

    def _store_sync(
        self,
        job: TransferJobState,
        plans: tuple[BlockTransferPlan, ...],
        ready_events: list[torch.Event],
        range_args: _StoreArgs | None = None,
    ) -> TransferJobState:
        # Timed from here: a store queued behind others has not stalled.
        self._store_deadlines[id(job)] = time.monotonic() + self._timeout_s
        for event in ready_events:
            event.synchronize()
        if not plans:
            job.complete()
            return job
        keys, object_sizes, pointers, sizes, offsets = (
            self._range_args(plans) if range_args is None else range_args
        )
        results = self.client.batch_put_ranges_from_ptr(
            keys, object_sizes, pointers, sizes, offsets
        )
        failed_indices = [index for index, ok in enumerate(results) if not ok]
        for attempt in range(_STORE_RETRIES):
            if not failed_indices:
                break
            self.client.flush()
            time.sleep(_STORE_RETRY_BACKOFF_S * (attempt + 1))
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
        if not self.client.flush():
            job.fail([plan.key for plan in plans], "MORI UMBP flush failed")
            return job
        completed = [plan.key for plan, ok in zip(plans, results, strict=True) if ok]
        failed = [plan.key for plan, ok in zip(plans, results, strict=True) if not ok]
        if completed:
            job.complete(completed)
        if failed:
            job.fail(failed, "MORI UMBP range store failed")
        return job

    def wait(self, job: TransferJobState) -> TransferJobState:
        future = self._futures.get(id(job))
        if future is None:
            return job
        try:
            return future.result(timeout=self._timeout_s)
        except TimeoutError as exc:
            if not future.done():
                # The transfer still owns GPU pointers. Abort this engine step
                # rather than reporting completion and allowing block reuse.
                raise TimeoutError(
                    "MORI UMBP transfer timed out with buffers still in use"
                ) from exc
            job.fail([plan.key for plan in job.plans], str(exc))
            return job
        except Exception as exc:
            job.fail([plan.key for plan in job.plans], str(exc))
            return job
        finally:
            if future.done():
                self._futures.pop(id(job), None)
                self._store_deadlines.pop(id(job), None)

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
            deadline = self._store_deadlines.get(id(job))
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "MORI UMBP transfer timed out with buffers still in use"
                )
            return None
        self._futures.pop(id(job), None)
        self._store_deadlines.pop(id(job), None)
        try:
            return future.result()
        except Exception as exc:
            job.fail([plan.key for plan in job.plans], str(exc))
            return job

    def publish(self, job: TransferJobState) -> None:
        if job.status.value != "completed":
            raise RuntimeError("cannot publish an incomplete MORI UMBP job")
        with self._key_lock:
            self._published_keys.update(job.completed_keys)

    def cancel(self, job: TransferJobState) -> TransferJobState:
        future = self._futures.get(id(job))
        if future is None:
            return job
        if future.cancel():
            self._futures.pop(id(job), None)
            self._store_deadlines.pop(id(job), None)
            job.cancel("preempted")
            return job
        return self.wait(job)

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


def _resolve_embedded_options(config: UMBPRuntimeConfig) -> dict[str, Any]:
    options = dict(config.options)
    if "capacity_bytes" in options and "total_capacity_bytes" in options:
        raise ValueError(
            "capacity_bytes and total_capacity_bytes are mutually exclusive"
        )
    for name in ("capacity_bytes", "total_capacity_bytes", "dram_hugepage_size"):
        if name in options and (type(options[name]) is not int or options[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("dram_use_shared_memory", "dram_use_hugepages", "dram_prefault"):
        if name in options and type(options[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    for name in ("dram_high_watermark", "dram_low_watermark"):
        if name in options and (
            type(options[name]) not in (int, float) or not 0 < options[name] <= 1
        ):
            raise ValueError(f"{name} must be in (0, 1]")
    low, high = options.get("dram_low_watermark"), options.get("dram_high_watermark")
    if low is not None and high is not None and low > high:
        raise ValueError("dram_low_watermark must not exceed dram_high_watermark")
    if "dram_numa_node" in options and (
        type(options["dram_numa_node"]) is not int or options["dram_numa_node"] < -1
    ):
        raise ValueError("dram_numa_node must be an integer >= -1")
    if "dram_shm_name" in options and (
        not isinstance(options["dram_shm_name"], str) or not options["dram_shm_name"]
    ):
        raise ValueError("dram_shm_name must be a non-empty string")
    if {
        "master_address",
        "node_address",
        "io_engine_host",
        "peer_service_port",
    } & options.keys():
        raise ValueError("embedded UMBP cannot configure distributed-only options")
    total = options.pop("total_capacity_bytes", None)
    if total is not None:
        if total < config.rank_count:
            raise ValueError(
                "total_capacity_bytes must provide at least one byte per rank"
            )
        options["capacity_bytes"] = total // config.rank_count
        options["_configured_total_capacity_bytes"] = total
    return options


class EmbeddedRuntime(IUMBPRuntime):
    """MORI-backed embedded DRAM runtime."""

    def __init__(self, config: UMBPRuntimeConfig) -> None:
        if config.options.get("backend") == "memory":
            raise ValueError(
                "The memory backend is test-only; use MORI for embedded mode"
            )
        self.options = _resolve_embedded_options(config)
        self.lookup_dir = self.options.get("lookup_dir", "/tmp")
        instance = self.options.get("lookup_instance", "")
        if not isinstance(instance, str):
            raise ValueError("lookup_instance must be a string")
        self.lookup_instance = f"{instance}:dp{config.dp_index}"

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        del layout
        return _MoriSchedulerHandle(
            namespace, topology, self.lookup_dir, self.lookup_instance
        )

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        try:
            from mori.cpp import UMBPClient, UMBPConfig
        except ImportError as exc:
            raise RuntimeError(
                "Embedded UMBP requires MORI built with BUILD_UMBP=ON"
            ) from exc

        client_config = UMBPConfig()
        _configure_dram(client_config, self.options)
        total = self.options.get("_configured_total_capacity_bytes")
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
        return _MoriWorkerHandle(
            UMBPClient(client_config),
            namespace,
            topology,
            self.lookup_dir,
            int(self.options.get("num_workers", 4)),
            float(self.options.get("timeout_ms", 30000)) / 1000,
            layout,
            lookup_instance=self.lookup_instance,
        )

    @classmethod
    def from_config(cls, config: UMBPRuntimeConfig) -> IUMBPRuntime:
        return cls(config)
