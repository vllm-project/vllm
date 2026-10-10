# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed UMBP runtime: one KV pool spanning every participating host.

A ``umbp_master`` indexes objects cluster-wide and routes puts. Every worker
is a pool node: it owns a slice of host DRAM, serves it to peers over MORI-IO
RDMA, and registers with the master under an identity of its own. Any engine
attached to the same master can therefore reuse KV that another engine stored,
on this host or another one.

Scheduler lookups go to the master through a client that has no pool, no
transfer engine, and no peer service: it only answers ``batch_exists``.
"""

from __future__ import annotations

import itertools
import math
import os
import socket
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import torch

from vllm.logger import init_logger

from ..data import KVLayoutDescriptor, RankTopology
from .base import (
    IUMBPRuntime,
    UMBPRuntimeCapabilities,
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from .embedded import _configure_dram, _split_total_capacity, _validate_dram_options
from .factory import UMBPRuntimeConfig
from .standalone import _SharedPoolWorkerHandle

logger = init_logger(__name__)

# A remote ranged transfer stages whole objects in a host arena, one for gets
# and one for puts, so each must hold at least one object. Larger arenas batch
# more objects per RDMA round.
DEFAULT_RANGED_SCRATCH_BYTES = 128 * 1024**2
# MORI master's default allocation granularity for DRAM objects.
_MASTER_DEFAULT_PAGE_BYTES = 2 * 1024**2
_PAGE_PADDING_WARN_RATIO = 0.25
# Measured on AMD Pollara (ionic): with 4 KiB pages, a 1 GiB pool plus the
# scratch arenas sometimes failed RDMA registration, and 2 GiB always did.
_LARGE_4K_POOL_BYTES = 512 * 1024**2
_LOOKUP_FAILURE_LOG_INTERVAL_S = 30.0
_client_sequence = itertools.count()
_LOCAL_ONLY_OPTIONS = frozenset(
    {"endpoint", "auto_start", "startup_timeout_ms", "lookup_dir"}
)
# Embedded DRAM options a distributed node ignores: eviction is decided by the
# master, and the pool is never shared through a named segment.
_IGNORED_DRAM_OPTIONS = frozenset(
    {
        "dram_use_shared_memory",
        "dram_shm_name",
        "dram_high_watermark",
        "dram_low_watermark",
    }
)


def _is_host_port(address: str) -> bool:
    host, sep, port = address.rpartition(":")
    return bool(sep and host and port.isdigit() and 0 < int(port) < 65536)


def _resolve_distributed_options(config: UMBPRuntimeConfig) -> dict[str, Any]:
    options = dict(config.options)
    address = options.get("master_address")
    if not isinstance(address, str) or not _is_host_port(address):
        raise ValueError("distributed UMBP requires master_address as 'host:port'")
    # Every worker serves its pool to peers; the port it binds is this base
    # plus its physical GPU index, so it must be set explicitly.
    if "peer_service_port" not in options:
        raise ValueError("distributed UMBP requires peer_service_port")
    for name in ("peer_service_port", "io_engine_port"):
        if name in options and (
            type(options[name]) is not int or not 0 < options[name] < 65536
        ):
            raise ValueError(f"{name} must be a port number")
    for name in ("node_address", "node_id", "io_engine_host", "backend_policy_path"):
        if name in options and (
            not isinstance(options[name], str) or not options[name]
        ):
            raise ValueError(f"{name} must be a non-empty string")
    for name in (
        "dram_page_size",
        "staging_buffer_size",
        "lookup_timeout_ms",
        "ranged_scratch_size",
    ):
        if name in options and (type(options[name]) is not int or options[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("cache_remote_fetches", "ranged_locality_prefetch", "local_first"):
        if name in options and type(options[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    local = sorted(_LOCAL_ONLY_OPTIONS.intersection(options))
    if local:
        raise ValueError(
            "distributed UMBP cannot configure " + ", ".join(local) + ", which "
            "only apply to the embedded and standalone modes"
        )
    ignored = sorted(_IGNORED_DRAM_OPTIONS.intersection(options))
    if ignored:
        raise ValueError(
            "distributed UMBP does not use " + ", ".join(ignored) + "; the "
            "master decides eviction and each worker owns a private pool"
        )
    _validate_dram_options(options)
    _split_total_capacity(options, config.rank_count)
    return options


def _node_id(prefix: str, role: str) -> str:
    """Return an identity no other live or recently dead process holds.

    The master refuses to re-register a node id until its previous holder has
    missed enough heartbeats to expire, so an id reused by a restarted engine
    would block that engine's startup for tens of seconds. Hostname and pid
    repeat across container restarts, hence the random suffix.
    """
    return (
        f"{prefix}-{socket.gethostname()}-{role}-{os.getpid()}-"
        f"{next(_client_sequence)}-{uuid.uuid4().hex[:8]}"
    )


def _physical_device_index() -> int:
    if not torch.accelerator.is_available():
        return 0
    from vllm.platforms import current_platform

    return int(
        current_platform.visible_device_id_to_physical_device_id(
            torch.accelerator.current_device_index()
        )
    )


def _node_address(options: dict[str, Any]) -> str:
    address = options.get("node_address")
    if address:
        return address
    from vllm.utils.network_utils import get_ip

    return get_ip()


def _build_client(
    options: dict[str, Any],
    *,
    node_id: str,
    pool: bool,
    peer_service_port: int = 0,
    io_engine_port: int = 0,
) -> Any:
    try:
        from mori.cpp import (
            UMBPClient,
            UMBPConfig,
            UMBPDeploymentMode,
            UMBPDistributedConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Distributed UMBP requires MORI built with BUILD_UMBP=ON"
        ) from exc

    master = options["master_address"]
    node_address = _node_address(options)
    config = UMBPConfig()
    dist = UMBPDistributedConfig()
    dist.master_config.master_address = master
    dist.master_config.node_id = node_id
    dist.master_config.node_address = node_address
    dist.master_config.auto_heartbeat = True
    if pool:
        _configure_dram(config, options)
        dist.io_engine.host = options.get("io_engine_host", node_address)
        dist.io_engine.port = io_engine_port
        dist.peer_service_port = peer_service_port
        dist.ranged_scratch_size = int(
            options.get("ranged_scratch_size", DEFAULT_RANGED_SCRATCH_BYTES)
        )
        for name in (
            "dram_page_size",
            "staging_buffer_size",
            "cache_remote_fetches",
            "ranged_locality_prefetch",
            "local_first",
            "backend_policy_path",
        ):
            if name in options:
                setattr(dist, name, options[name])
    else:
        # A zero-sized pool registers no tier, so the master never routes a
        # put here, and with no I/O engine or peer service nothing is served.
        config.dram.capacity_bytes = 0
        dist.cache_remote_fetches = False
        dist.ranged_locality_prefetch = False
    config.distributed = dist

    try:
        client = UMBPClient(config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot join the UMBP master at {master} as {node_id} "
            f"(node_address={node_address}): {exc}"
        ) from exc
    mode = client.get_deployment_mode()
    if mode != UMBPDeploymentMode.Distributed:
        raise RuntimeError(
            f"UMBP client for master {master} reports deployment mode {mode}, "
            "expected Distributed"
        )
    return client


class _DistributedSchedulerHandle(UMBPSchedulerHandle):
    """Answers lookups from the master, which sees every node's objects.

    MORI reports a master that refuses connections as all-miss, but its lookup
    RPC has no deadline: against a master that accepts and never answers, it
    blocks until the master recovers. Lookups run on the scheduler's critical
    path, so each one is bounded and reported as all-miss when it runs over.
    While one is still blocked, later lookups are answered as misses without
    asking the master, so an outage costs one timeout rather than one per
    step.
    """

    def __init__(self, client: Any, master: str, timeout_s: float) -> None:
        self._client = client
        self._master = master
        self._timeout_s = timeout_s
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="umbp-master-lookup"
        )
        self._lock = threading.Lock()
        self._blocked: Future[Any] | None = None
        self._last_failure_log = float("-inf")

    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        if not keys:
            return []
        with self._lock:
            if self._blocked is not None:
                if not self._blocked.done():
                    self._log_failure(
                        TimeoutError("an earlier lookup is still waiting on it")
                    )
                    return [False] * len(keys)
                self._blocked = None
            future = self._executor.submit(self._client.batch_exists, list(keys))
            try:
                values = future.result(timeout=self._timeout_s)
            except FutureTimeoutError:
                self._blocked = future
                self._log_failure(
                    TimeoutError(f"no answer within {self._timeout_s:.1f}s")
                )
                return [False] * len(keys)
            except Exception as exc:
                self._log_failure(exc)
                return [False] * len(keys)
        result = [bool(value) for value in values]
        if len(result) != len(keys):
            self._log_failure(ValueError(f"{len(result)} results for {len(keys)} keys"))
            return [False] * len(keys)
        return result

    def clear(self) -> bool:
        # Clearing through one engine would drop KV every other engine on the
        # cluster relies on, and MORI can only clear the caller's own node,
        # which for this client is empty. Report that nothing was cleared.
        logger.warning(
            "UMBP distributed pool behind master %s is shared by every engine "
            "attached to it and is not cleared by resetting one engine",
            self._master,
        )
        return False

    def close(self) -> None:
        # A lookup blocked on the master must not hold up engine shutdown.
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._client = None

    def _log_failure(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_failure_log >= _LOOKUP_FAILURE_LOG_INTERVAL_S:
            self._last_failure_log = now
            logger.warning(
                "UMBP distributed lookup against master %s failed; treating as "
                "a miss: %s",
                self._master,
                exc,
            )


class _DistributedWorkerHandle(_SharedPoolWorkerHandle):
    mode = "distributed"


class DistributedRuntime(IUMBPRuntime):
    """Runtime whose objects live in pool nodes coordinated by a master."""

    capabilities = UMBPRuntimeCapabilities(
        layerwise_load=True,
        partial_hash_hits=True,
    )

    def __init__(self, config: UMBPRuntimeConfig) -> None:
        options = _resolve_distributed_options(config)
        self._options = options
        self._master = options["master_address"]
        self._prefix = options.get("node_id", "vllm")
        self._max_workers = int(options.get("num_workers", 4))
        self._timeout_s = float(options.get("timeout_ms", 30000)) / 1000
        self._lookup_timeout_s = float(options.get("lookup_timeout_ms", 2000)) / 1000

    @classmethod
    def from_config(cls, config: UMBPRuntimeConfig) -> DistributedRuntime:
        return cls(config)

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        del namespace, topology, layout
        client = _build_client(
            self._options, node_id=_node_id(self._prefix, "lookup"), pool=False
        )
        return _DistributedSchedulerHandle(client, self._master, self._lookup_timeout_s)

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        options = self._options
        scratch = int(options.get("ranged_scratch_size", DEFAULT_RANGED_SCRATCH_BYTES))
        object_size = layout.object_size if layout is not None else 0
        if object_size > scratch:
            raise ValueError(
                f"UMBP ranged_scratch_size ({scratch} bytes) is smaller than one "
                f"KV object ({object_size} bytes); every transfer to or from "
                "another node would fail"
            )
        if object_size:
            _warn_on_page_padding(object_size, options)
        capacity = options.get("capacity_bytes", 64 * 1024**3)
        if capacity > _LARGE_4K_POOL_BYTES and not options.get("dram_use_hugepages"):
            logger.warning(
                "UMBP distributed pool of %.1f GiB uses 4 KiB pages. NICs with a "
                "small translation cache (e.g. AMD Pollara) cannot register that "
                "much 4 KiB-paged memory for RDMA, and MORI aborts the process "
                "when a registration fails; reserve hugepages "
                "(vm.nr_hugepages) and set dram_use_hugepages=true",
                capacity / 1024**3,
            )
        # Several workers share a host, and peers dial each one directly, so
        # each binds the configured base port offset by its physical GPU.
        device = _physical_device_index()
        peer_port = options["peer_service_port"] + device
        io_port = (
            options["io_engine_port"] + device if "io_engine_port" in options else 0
        )
        if max(peer_port, io_port) > 65535:
            raise ValueError(
                f"UMBP port base plus physical GPU {device} exceeds 65535; lower "
                "peer_service_port or io_engine_port"
            )
        node_id = _node_id(self._prefix, f"gpu{device}")
        try:
            client = _build_client(
                options,
                node_id=node_id,
                pool=True,
                peer_service_port=peer_port,
                io_engine_port=io_port,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}. Each worker binds peer_service_port + its physical GPU "
                f"index ({peer_port} here); two workers seeing the same GPU "
                "index on one host, e.g. one GPU per container, collide"
            ) from exc
        logger.info(
            "UMBP distributed worker %s joined master %s: %.2f GiB pool, "
            "peer service port %d, tp_rank %d",
            node_id,
            self._master,
            options.get("capacity_bytes", 64 * 1024**3) / 1024**3,
            peer_port,
            topology.tp_rank,
        )
        return _DistributedWorkerHandle(
            client,
            namespace,
            topology,
            lookup_dir="",
            max_workers=self._max_workers,
            timeout_s=self._timeout_s,
            layout=layout,
            serve_lookups=False,
        )


def _warn_on_page_padding(object_size: int, options: dict[str, Any]) -> None:
    page = options.get("dram_page_size", _MASTER_DEFAULT_PAGE_BYTES)
    padded = math.ceil(object_size / page) * page
    if padded - object_size > _PAGE_PADDING_WARN_RATIO * object_size:
        logger.warning(
            "UMBP KV objects are %d bytes but pool pages are %d bytes, so each "
            "object carries %.0f%% padding; set dram_page_size to a divisor of the "
            "object size to use it fully",
            object_size,
            page,
            100 * (padded - object_size) / object_size,
        )
