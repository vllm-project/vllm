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

import time
from collections.abc import Sequence
from typing import Any

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
    if "startup_timeout_ms" in options and (
        type(options["startup_timeout_ms"]) is not int
        or options["startup_timeout_ms"] <= 0
    ):
        raise ValueError("startup_timeout_ms must be a positive integer")
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

    def _blocked_stores(self) -> int:
        now = time.monotonic()
        return sum(
            1
            for job_id, deadline in self._store_deadlines.items()
            if now >= deadline
            and (future := self._futures.get(job_id)) is not None
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
        future = self._futures.get(id(job))
        if future is not None and not future.done():
            return None
        return super().poll(job)

    def cancel(self, job: TransferJobState) -> TransferJobState:
        if id(job) in self._store_deadlines:
            future = self._futures.get(id(job))
            if future is not None and future.cancel():
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
        # A call still blocked in MORI must not hold up the shutdown.
        blocked = any(not future.done() for future in self._futures.values())
        self._executor.shutdown(wait=not blocked, cancel_futures=True)
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
        # MORI releases the connection only when the client object is destroyed.
        self.client = None  # type: ignore[assignment]


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
