# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone UMBP runtime backed by a node-local ``umbp_standalone_server``.

The server owns one DRAM pool per host. Every engine, data-parallel rank, and
worker on that host attaches to it over a Unix-domain socket, so KV stored by
one engine is reusable by the others and survives engine restarts. Workers
register their GPU KV allocations with the server through HIP IPC; the server
then copies between those allocations and its pool directly.

Because the server is reachable from the scheduler process, scheduler lookups
query it directly instead of going through a worker-side bridge.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future
from concurrent.futures import wait as wait_futures
from typing import Any

import torch

from vllm.logger import init_logger

from ..data import (
    TRANSFER_NOT_ATTEMPTED,
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
)
from .base import IUMBPRuntime, UMBPSchedulerHandle, UMBPWorkerHandle
from .embedded import _MoriWorkerHandle, _resolve_embedded_options
from .factory import UMBPRuntimeConfig

logger = init_logger(__name__)

_UNIX_SCHEME = "unix://"
_FAILURE_LOG_INTERVAL_S = 30.0

# Registered KV storages of a client left open at shutdown. The service may
# still read or write them once it resumes, so they live until process exit.
_ABANDONED_STORAGES: list[torch.UntypedStorage] = []


class _DaemonTransferPool(Executor):
    """A thread pool whose threads never hold up interpreter exit.

    ThreadPoolExecutor joins its threads at exit, and a call blocked in an
    unresponsive service cannot be interrupted.
    """

    def __init__(self, max_workers: int, thread_name_prefix: str) -> None:
        self._work: queue.SimpleQueue[tuple[Future, Callable[[], Any]] | None] = (
            queue.SimpleQueue()
        )
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads = [
            threading.Thread(
                target=self._run, name=f"{thread_name_prefix}_{index}", daemon=True
            )
            for index in range(max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._work.put((future, lambda: fn(*args, **kwargs)))
        return future

    def _run(self) -> None:
        while (item := self._work.get()) is not None:
            future, call = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = call()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            if cancel_futures:
                while True:
                    try:
                        item = self._work.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        item[0].cancel()
            for _ in self._threads:
                self._work.put(None)
        if wait:
            for thread in self._threads:
                thread.join()


def normalize_endpoint(endpoint: str) -> str:
    """Return the gRPC address of a standalone server socket.

    The data plane depends on file-descriptor and IPC-handle passing over a
    Unix-domain socket, so only socket paths are accepted.
    """
    if endpoint.startswith(_UNIX_SCHEME):
        path = endpoint[len(_UNIX_SCHEME) :]
    elif "://" in endpoint:
        raise ValueError(
            "standalone UMBP endpoint must be a Unix socket path or unix:// "
            f"address, got {endpoint!r}"
        )
    else:
        path = endpoint
    if not path.startswith("/"):
        raise ValueError(
            f"standalone UMBP endpoint must be an absolute path, got {endpoint!r}"
        )
    return _UNIX_SCHEME + path


def _resolve_standalone_options(config: UMBPRuntimeConfig) -> dict[str, Any]:
    options = config.options
    endpoint = options.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("standalone UMBP requires endpoint")
    normalize_endpoint(endpoint)
    for name in ("startup_timeout_ms", "lookup_timeout_ms"):
        if name in options and (type(options[name]) is not int or options[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    server_options = sorted(
        name
        for name in options
        if name in ("auto_start", "capacity_bytes", "total_capacity_bytes")
        or name.startswith("dram_")
    )
    if server_options:
        raise ValueError(
            "standalone UMBP attaches to a server that owns its DRAM pool; "
            + ", ".join(server_options)
            + " do not apply: launch umbp_standalone_server and configure "
            "its UMBP_DRAM_* environment"
        )
    return _resolve_embedded_options(config)


def _build_client(options: dict[str, Any]) -> Any:
    try:
        from mori.cpp import (
            UMBPClient,
            UMBPConfig,
            UMBPDeploymentMode,
            UMBPStandaloneProcessConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Standalone UMBP requires MORI built with BUILD_UMBP=ON"
        ) from exc

    address = normalize_endpoint(options["endpoint"])
    client_config = UMBPConfig()
    standalone = UMBPStandaloneProcessConfig()
    standalone.address = address
    standalone.auto_start = False
    standalone.startup_timeout_ms = int(options.get("startup_timeout_ms", 30000))
    if hasattr(standalone, "lookup_timeout_ms"):
        standalone.lookup_timeout_ms = int(options.get("lookup_timeout_ms", 1000))
    else:
        logger.warning_once(
            "MORI bindings do not support lookup_timeout_ms; ignoring it, so "
            "standalone UMBP lookups have no native deadline"
        )
    client_config.standalone_process = standalone

    try:
        client = UMBPClient(client_config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot attach to the UMBP standalone server at {address}; start "
            "umbp_standalone_server on this host first"
        ) from exc
    mode = client.get_deployment_mode()
    if mode != UMBPDeploymentMode.StandaloneProcess:
        raise RuntimeError(
            f"UMBP client for {address} reports deployment mode {mode}, "
            "expected StandaloneProcess"
        )
    return client


class _StandaloneSchedulerHandle(UMBPSchedulerHandle):
    """Answers lookups from the shared server.

    A lookup failure is reported as a miss so an unreachable server degrades
    to recomputation instead of failing the scheduler step.
    """

    def __init__(self, client: Any, address: str) -> None:
        self._client = client
        self._address = address
        self._last_failure_log = float("-inf")

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        if not keys:
            return []
        try:
            result = [bool(value) for value in self._client.batch_exists(list(keys))]
        except Exception as exc:
            self._log_failure("lookup", exc)
            return [False] * len(keys)
        if len(result) != len(keys):
            self._log_failure(
                "lookup", ValueError(f"{len(result)} results for {len(keys)} keys")
            )
            return [False] * len(keys)
        return result

    def clear(self) -> bool:
        try:
            cleared = bool(self._client.clear())
        except Exception as exc:
            self._log_failure("clear", exc)
            return False
        if cleared:
            logger.warning(
                "Cleared the UMBP standalone pool at %s; this removes KV for "
                "every engine attached to it",
                self._address,
            )
        else:
            logger.warning("UMBP standalone clear at %s failed", self._address)
        return cleared

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        try:
            if callable(close):
                close()
        except Exception as exc:
            logger.warning("UMBP standalone client close failed: %s", exc)
        self._client = None

    def _log_failure(self, operation: str, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_failure_log >= _FAILURE_LOG_INTERVAL_S:
            self._last_failure_log = now
            logger.warning(
                "UMBP standalone %s against %s failed; treating as a miss: %s",
                operation,
                self._address,
                exc,
            )


class _SharedPoolWorkerHandle(_MoriWorkerHandle):
    """MORI worker handle whose objects are indexed outside this process.

    The pool's service can stall or disappear under a running engine, so its
    failures degrade to misses and recomputation. A store that outlives
    timeout_ms stays pending rather than failing the step: the scheduler keeps
    its source blocks pinned until MORI returns, since MORI still reads them.
    """

    mode = "shared"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_refusal_log = float("-inf")
        self._storages: dict[int, torch.UntypedStorage] = {}

    def _create_executor(self, max_workers: int) -> Executor:
        return _DaemonTransferPool(max_workers, f"umbp-{self.mode}-transfer")

    def register_buffers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_buffers(kv_caches)
        for cache in kv_caches.values():
            storage = cache.untyped_storage()
            self._storages.setdefault(storage.data_ptr(), storage)

    def _blocked_stores(self) -> int:
        now = time.monotonic()
        with self._job_lock:
            deadlines = tuple(self._store_deadlines.items())
            futures = dict(self._futures)
        return sum(
            1
            for job_id, deadline in deadlines
            if now >= deadline
            and (future := futures.get(job_id)) is not None
            and not future.done()
        )

    def load(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        refused = self._refuse_while_blocked(plans)
        return refused if refused is not None else super().load(plans)

    def load_blocks(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        refused = self._refuse_while_blocked(plans)
        return refused if refused is not None else super().load_blocks(plans)

    def store(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        refused = self._refuse_while_blocked(plans)
        return refused if refused is not None else super().store(plans)

    def store_blocks(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        refused = self._refuse_while_blocked(plans)
        return refused if refused is not None else super().store_blocks(plans)

    def poll(self, job: TransferJobState) -> TransferJobState | None:
        with self._job_lock:
            future = self._futures.get(id(job))
            is_load = id(job) in self._load_deadlines
        if future is not None and not future.done():
            if self._blocked_stores() and self._fail_if_queued(job):
                return job
            if not is_load:
                return None
        return super().poll(job)

    def wait(self, job: TransferJobState) -> TransferJobState:
        try:
            return super().wait(job)
        except TimeoutError:
            if self._fail_if_queued(job):
                return job
            raise

    def _fail_if_queued(self, job: TransferJobState) -> bool:
        """Fail a job still waiting for a transfer thread; it holds no buffers."""
        with self._job_lock:
            future = self._futures.get(id(job))
        if future is None or not future.cancel():
            return False
        with self._job_lock:
            self._futures.pop(id(job), None)
        job.fail(
            job.keys,
            f"{TRANSFER_NOT_ATTEMPTED}: queued behind a blocked UMBP {self.mode} store",
        )
        return True

    def cancel(self, job: TransferJobState) -> TransferJobState:
        with self._job_lock:
            is_store = id(job) in self._store_deadlines
            future = self._futures.get(id(job))
        if is_store:
            if future is not None and future.cancel():
                with self._job_lock:
                    self._futures.pop(id(job), None)
                    self._store_deadlines.pop(id(job), None)
                job.cancel("preempted")
            # A running store only reads blocks the scheduler keeps pinned.
            return job
        return super().cancel(job)

    def _refuse_while_blocked(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        """Fail new transfers, before they touch memory, while a store is stuck."""
        blocked = self._blocked_stores()
        if not blocked:
            return None
        job = TransferJobState(
            plans if isinstance(plans, BlockLoadBatch) else tuple(plans)
        )
        job.start()
        job.fail(
            job.keys,
            f"{TRANSFER_NOT_ATTEMPTED}: an earlier UMBP {self.mode} store "
            "is still blocked",
        )
        now = time.monotonic()
        if now - self._last_refusal_log >= _FAILURE_LOG_INTERVAL_S:
            self._last_refusal_log = now
            logger.warning(
                "UMBP %s transfers are refused while %d timed-out store(s) are "
                "still blocked in MORI",
                self.mode,
                blocked,
            )
        return job

    def publish(self, job: TransferJobState) -> None:
        if job.status != TransferJobStatus.COMPLETED:
            raise RuntimeError("cannot publish an incomplete MORI UMBP job")
        try:
            super().publish(job)
        except Exception as exc:
            logger.warning("UMBP %s publish failed: %s", self.mode, exc)

    def close(self) -> None:
        # The service may already be gone at engine shutdown and its
        # registrations with it, so a failed deregistration is not an error.
        with self._job_lock:
            futures = list(self._futures.values())
            deadlines = [
                *self._store_deadlines.values(),
                *self._load_deadlines.values(),
            ]
        self._executor.shutdown(wait=False, cancel_futures=True)
        # Running transfers get until their own deadlines; stalled ones get none.
        timeout = max([0.0, *(deadline - time.monotonic() for deadline in deadlines)])
        _, in_flight = wait_futures(futures, timeout=timeout)
        if in_flight:
            # MORI's deregister_memory() and close() wait for every in-flight
            # call, and a stalled service never returns one.
            self._abandon(f"{len(in_flight)} transfer(s) are still blocked in MORI")
            return
        # Neither MORI call has a deadline, so an idle client can hang on them too.
        release = threading.Thread(
            target=self._release_client, name=f"umbp-{self.mode}-close", daemon=True
        )
        release.start()
        release.join(self._timeout_s)
        if release.is_alive():
            self._abandon(f"deregistration did not return in {self._timeout_s:g}s")

    def _release_client(self) -> None:
        for storage_ptr in self._registered_storages:
            try:
                self.client.deregister_memory(storage_ptr)
            except Exception as exc:
                logger.warning(
                    "UMBP %s deregister_memory(0x%x) failed: %s",
                    self.mode,
                    storage_ptr,
                    exc,
                )
        self._registered_storages.clear()
        close = getattr(self.client, "close", None)
        try:
            if callable(close):
                close()
        except Exception as exc:
            logger.warning("UMBP %s client close failed: %s", self.mode, exc)
        self._storages.clear()
        # MORI releases the connection only when the client object is destroyed.
        self.client = None  # type: ignore[assignment]

    def _abandon(self, reason: str) -> None:
        _ABANDONED_STORAGES.extend(self._storages.values())
        self._storages.clear()
        logger.warning("UMBP %s client left open at shutdown: %s", self.mode, reason)


class _StandaloneWorkerHandle(_SharedPoolWorkerHandle):
    mode = "standalone"


class StandaloneRuntime(IUMBPRuntime):
    """Runtime whose objects live in a node-local standalone server."""

    def __init__(self, config: UMBPRuntimeConfig) -> None:
        self._options = _resolve_standalone_options(config)
        self._address = normalize_endpoint(self._options["endpoint"])
        self._max_workers = int(self._options.get("num_workers", 4))
        self._timeout_s = float(self._options.get("timeout_ms", 30000)) / 1000

    @classmethod
    def from_config(cls, config: UMBPRuntimeConfig) -> StandaloneRuntime:
        return cls(config)

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        del namespace, topology, layout
        return _StandaloneSchedulerHandle(_build_client(self._options), self._address)

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        logger.info("UMBP standalone worker attached to %s", self._address)
        return _StandaloneWorkerHandle(
            _build_client(self._options),
            namespace,
            topology,
            lookup_dir="",
            max_workers=self._max_workers,
            timeout_s=self._timeout_s,
            layout=layout,
            serve_lookups=False,
        )
