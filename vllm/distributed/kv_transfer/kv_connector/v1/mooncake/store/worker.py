# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The transfer-thread scaffolding (KVTransferThread, KVCacheStoreSendingThread,
# KVCacheStoreRecvingThread) is adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Worker-side logic for MooncakeStoreConnector.

Includes the store worker, transfer threads, lookup server,
and MooncakeDistributedStore integration.
"""

import dataclasses
import json
import os
import queue
import socket
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import regex as re
import torch
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    BlobBlockHashes,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerTransferTask,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    PoolKey,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # noqa: E501
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from .metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_TENANT_ID = "default"

MOONCAKE_NO_AVAILABLE_HANDLE = -200
_T = TypeVar("_T")

# Mooncake Session API methods (PR#2881) — used for capability detection.
_MOONCAKE_SESSION_METHODS = (
    "batch_put_session_start",
    "batch_put_from_multi_buffer_ranges",
    "batch_put_session_end",
    "batch_put_session_revoke",
    "batch_get_session_start",
    "batch_get_into_multi_buffer_ranges",
    "batch_get_session_end",
)


def _mooncake_supports_session_api(store: Any) -> bool:
    """Return True if the Mooncake store exposes all Session API methods.

    Uses ``hasattr`` (which maps to ``__getattr__`` + catch), not ``getattr``,
    because the real Mooncake client will raise AttributeError for unknown
    methods rather than silently returning None.
    """
    for method in _MOONCAKE_SESSION_METHODS:
        try:
            attr = getattr(store, method)
        except Exception:
            return False
        if not callable(attr):
            return False
    return True


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]


def _replicate_config_supports_group_ids(
    replicate_config_cls: type[Any],
    replicate_config: Any,
) -> bool:
    if hasattr(replicate_config_cls, "group_ids"):
        return True
    return hasattr(replicate_config, "group_ids")


def _make_mooncake_group_id(metadata: KeyMetadata, chunk_hash: str) -> str:
    # Mooncake group ids describe the lifecycle unit. For vLLM, that unit is
    # a prefix chunk, so shard dimensions stay only in the object key.
    prefix = f"{metadata.cache_prefix}@" if metadata.cache_prefix else ""
    return f"vllm-mooncake-store:{prefix}{metadata.model_name}@{chunk_hash}"


# Mirrors FileStorageConfig::local_buffer_size in Mooncake C++.
DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES = 1280 * 1024 * 1024

# Mirrors DirectIO alignment in Mooncake's AllocateBatch.
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


MooncakeMode = Literal["embedded", "standalone-store"]


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    ``mode`` selects the topology: ``embedded`` (each rank contributes
    ``global_segment_size`` in-process) or ``standalone-store`` (rank
    contributes 0; an external ``mooncake_client`` process owns the pool
    and the SSD tier).
    """

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False
    tenant_id: str = DEFAULT_TENANT_ID

    def __post_init__(self) -> None:
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError("embedded mode requires global_segment_size > 0")
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            master_server_address=config.get("master_server_address", ""),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            mode=config.get("mode", "embedded"),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            enable_offload=bool(config.get("enable_offload", False)),
            tenant_id=_normalize_tenant_id(config.get("tenant_id", DEFAULT_TENANT_ID)),
        )

    @staticmethod
    def load_from_config() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


def _normalize_tenant_id(value: Any) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    if not isinstance(value, str):
        raise TypeError(
            f"tenant_id must be a string or null, got {type(value).__name__}: {value!r}"
        )
    tenant_id = value.strip()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


def _parse_size(value: Any) -> int:
    """Parse storage size strings with units: GB, MB, KB, B."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for size: {type(value)}") from e

    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("Size cannot be empty.")

    unit_multipliers = {
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "b": 1,
    }
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"
    multiplier = unit_multipliers[unit]

    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{value}'") from exc
    return int(numeric_value * multiplier)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    data_size = sum(size_list)
    return _align_up(data_size, _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _sum_batch_bytes(sizes: list[list[int]]) -> int:
    return sum(sum(size) for size in sizes)


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    return max(1, int(raw_budget_bytes * envs.VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    """Split a GET into sub-batches that fit the owner's staging buffer.

    ``addrs[i]`` / ``sizes[i]`` are scatter-gather lists (K/V or multi-layer
    segments) for key ``i``. ``usable_budget_bytes`` caps a multi-key batch;
    ``raw_budget_bytes`` is the hard per-key cap.

    Returns ``(batches, oversize_key)``. Aborts with ``([], key)`` if any
    single key exceeds ``raw_budget_bytes``; otherwise ``oversize_key`` is
    ``None``.
    """
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)
        if key_bytes > raw_budget_bytes:
            return [], key
        if key_bytes > usable_budget_bytes:
            if batch_keys:
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            batches.append(([key], [addr], [size]))
            continue
        if batch_keys and batch_bytes + key_bytes > usable_budget_bytes:
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


def _call_replica_predicate(replica_desc: Any, method_name: str) -> bool:
    method = getattr(replica_desc, method_name, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _classify_replica_tier(replica_descs: Any) -> str:
    if not replica_descs:
        return "unknown"
    try:
        replica_desc = replica_descs[0]
    except (IndexError, KeyError, TypeError):
        return "unknown"

    if _call_replica_predicate(replica_desc, "is_memory_replica"):
        return "memory"
    if _call_replica_predicate(
        replica_desc, "is_disk_replica"
    ) or _call_replica_predicate(replica_desc, "is_local_disk_replica"):
        return "disk"
    return "unknown"


def _get_replica_tiers_by_key(store: Any, keys: list[str]) -> dict[str, str]:
    tiers_by_key = {key: "unknown" for key in keys}
    try:
        replica_descs_by_key = store.batch_get_replica_desc(keys)
    except Exception as e:
        logger.warning(
            "Failed to get Mooncake replica descriptors for tier logging "
            "(batch_keys=%d, error=%s); marking tiers unknown",
            len(keys),
            e,
        )
        return tiers_by_key

    for key in keys:
        if hasattr(replica_descs_by_key, "get"):
            replica_descs = replica_descs_by_key.get(key)
        else:
            try:
                replica_descs = replica_descs_by_key[key]
            except (KeyError, TypeError):
                replica_descs = None
        tiers_by_key[key] = _classify_replica_tier(replica_descs)
    return tiers_by_key


def _log_mooncake_load_tier_summary(
    req_id: str,
    batch_keys: list[str],
    load_results: list[int],
    tiers_by_key: dict[str, str],
) -> None:
    tier_counts = {"memory": 0, "disk": 0, "unknown": 0}
    bytes_by_tier = {"memory": 0, "disk": 0, "unknown": 0}
    success_keys = 0
    failed_keys = 0

    for index, key in enumerate(batch_keys):
        tier = tiers_by_key.get(key, "unknown")
        if tier not in tier_counts:
            tier = "unknown"
        tier_counts[tier] += 1

        value = load_results[index] if index < len(load_results) else -1
        if value >= 0:
            success_keys += 1
            bytes_by_tier[tier] += int(value)
        else:
            failed_keys += 1

    logger.info(
        "Mooncake load tier summary: req_id=%s batch_keys=%d "
        "memory_keys=%d disk_keys=%d unknown_keys=%d "
        "success_keys=%d failed_keys=%d bytes_by_tier=%s",
        req_id,
        len(batch_keys),
        tier_counts["memory"],
        tier_counts["disk"],
        tier_counts["unknown"],
        success_keys,
        failed_keys,
        bytes_by_tier,
    )


# ============================================================
# Transfer Threads
# ============================================================


class KVTransferThread(threading.Thread):
    """Base class for async KV cache transfer threads.

    Extended with layerwise support for per-layer KV cache transfer.
    """

    _num_layers: int = 0
    _use_session_api: bool = False

    def __init__(
        self,
        store: Any,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self._record_operation_cb = record_operation
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = request_queue or queue.Queue()
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

        # Layerwise support
        self._layerwise_enabled = False
        self._layer_save_finished_events: dict[int, threading.Event] = {}
        self._layer_load_finished_events: dict[int, threading.Event] = {}

    def enable_layerwise(self, num_layers: int) -> None:
        """Enable layerwise mode with specified number of layers."""
        self._layerwise_enabled = True
        self._num_layers = num_layers
        for layer_id in range(num_layers):
            self._layer_save_finished_events[layer_id] = threading.Event()
            self._layer_load_finished_events[layer_id] = threading.Event()

    def set_layer_finished_event(
        self, layer_id: int, is_save: bool, event: threading.Event
    ) -> None:
        """Set the finished event for a specific layer."""
        if is_save:
            self._layer_save_finished_events[layer_id] = event
        else:
            self._layer_load_finished_events[layer_id] = event

    def add_request(self, request: ReqMeta | LayerTransferTask) -> None:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished

    def set_finished_request(self, req_id: str):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        self.ready_event.set()
        while True:
            request_data = None
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception:
                req_id = getattr(request_data, "req_id", "<unknown>")
                logger.exception("Error in %s (req=%s)", self.name, req_id)

    def _handle_request(self, req_meta: Any):
        pass

    def _record_operation(
        self,
        operation: str,
        start_time: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        if self._record_operation_cb is None:
            return
        self._record_operation_cb(
            operation=operation,
            duration_seconds=time.perf_counter() - start_time,
            num_keys=num_keys,
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

    def update_kv_event(self, events: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(events)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    """Background thread for storing KV cache blocks to the store."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        group_put_steps: Sequence[int],
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
        replicate_config: Any = None,
        enable_group_semantics: bool = False,
        supports_group_ids: bool = False,
        record_operation: Callable[..., None] | None = None,
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
            record_operation=record_operation,
        )
        # Only ranks with identical group bytes may stripe PUTs (e.g., MLA).
        self.group_put_steps = group_put_steps
        self.coord = coord
        self.kv_role = kv_role
        # req_id -> ids of its store jobs that are still queued or running.
        # Keying by store_job_id, which never repeats for the engine's lifetime,
        # rather than counting jobs per request id makes the ledger immune to id
        # reuse across preemption: a job left over from a retired generation is
        # missing from the set its resumed generation builds, so it can no longer
        # retire that generation, rewind its resume offset, or mark it skipped.
        self.stored_requests: dict[str, set[int]] = {}
        # store_job_id -> times this rank finished with it, drained every step
        # so the scheduler can release the blocks it referenced for those jobs.
        self._completed_saves: dict[int, int] = {}
        self.enable_kv_event = enable_kv_event
        # Caller always passes a non-None ReplicateConfig — see
        # MooncakeStoreWorker.__init__ where store_replicate_config is built.
        self.replicate_config = replicate_config
        self.enable_group_semantics = enable_group_semantics
        self.supports_group_ids = supports_group_ids

        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()

        # Per-request high-water mark of tokens actually persisted; the next
        # batch resumes here, so pressure-skipped or failed ranges are retried.
        self._saved_offset: dict[str, int] = {}

        # Session API state
        # Keys whose put session opened this step (set by start_put_sessions,
        # consumed by the per-layer range-put handler). None/empty until then.
        self._active_put_keys: set[str] | None = None

    def add_request(self, request: ReqMeta | LayerTransferTask) -> None:
        # Layerwise per-layer tasks carry no store_job_id; enqueue them directly
        # (the ledger only tracks bulk, non-layerwise store jobs).
        if isinstance(request, LayerTransferTask):
            super().add_request(request)
            return
        # Register before enqueueing so a job is never picked up unledgered.
        assert request.store_job_id is not None
        with self.done_task_lock:
            self.stored_requests.setdefault(request.req_id, set()).add(
                request.store_job_id
            )
        super().add_request(request)

    def is_live_store_job(self, req_meta: ReqMeta) -> bool:
        with self.done_task_lock:
            return req_meta.store_job_id in self.stored_requests.get(
                req_meta.req_id, ()
            )

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._skip_store_requests.discard(req_id)
            self._saved_offset.pop(req_id, None)

    def finish_store_job(self, req_meta: ReqMeta) -> None:
        """Retire a job from the ledger and report its blocks as no longer read.

        Every path out of a job must reach this, skips and failures included: a
        job that never reports leaves its blocks referenced for the rest of the
        run. The discard is a no-op for a job whose generation already retired.
        """
        store_job_id = req_meta.store_job_id
        assert store_job_id is not None, (
            "a queued store job always carries a store_job_id"
        )
        with self.done_task_lock:
            live = self.stored_requests.get(req_meta.req_id)
            if live is not None:
                live.discard(store_job_id)
            self._completed_saves[store_job_id] = (
                self._completed_saves.get(store_job_id, 0) + 1
            )

    def take_completed_saves(self) -> dict[int, int]:
        with self.done_task_lock:
            completed = self._completed_saves
            self._completed_saves = {}
        return completed

    def _record_saved(self, req_meta: ReqMeta, token_len: int) -> None:
        # Guard on job liveness so neither a concurrent finish/preempt pop nor a
        # stale job's offset is written back over the live generation's.
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._saved_offset[req_meta.req_id] = token_len

    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _mark_request_skipped_for_pressure(self, req_meta: ReqMeta) -> bool:
        req_id = req_meta.req_id
        with self.done_task_lock:
            already_skipped = req_id in self._skip_store_requests
            self._store_pressure_active = True
            # The pressure itself is global, but only a live job may sentence its
            # own request to being skipped.
            if req_meta.store_job_id in self.stored_requests.get(req_id, ()):
                self._skip_store_requests.add(req_id)
        return already_skipped

    def _clear_store_pressure(self) -> bool:
        with self.done_task_lock:
            if not self._store_pressure_active and not self._skip_store_requests:
                return False
            self._store_pressure_active = False
            self._skip_store_requests.clear()
        return True

    def _maybe_offload_partial_tail(self, req_meta: ReqMeta) -> bool:
        """Offload the request's sub-block partial tail (its last prompt hash
        boundary) so a later request can hit the sub-block prefix.

        Covers every block from the normal save's lcm floor to the boundary:
        the normal save floors to ``lcm_block_size``, so a smaller-block
        group's full blocks in that gap are never persisted elsewhere, and
        the consumer's lookup needs every group at every probed boundary.
        Full blocks are keyed by their block-end hash, the partial boundary
        block by the boundary sub-hash; the mamba "align" boundary block is
        the core-provided CoW block. All keys are deduped against the store.

        Returns:
            True when no put is needed or every put succeeds, False otherwise.
        """
        if not self.coord.enable_partial_hash_hits or not req_meta.block_hashes:
            return True
        partial_tail_offloads = req_meta.partial_tail_offloads
        if not partial_tail_offloads:
            return True
        hash_block_size = self.coord.hash_block_size
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError(
                "Partial-tail offloads for one request must share a boundary"
            )
        boundary = boundaries.pop()
        if boundary == 0:
            return True
        if boundary // hash_block_size - 1 >= len(req_meta.block_hashes):
            return True
        mamba_offloads = {
            group_id: block_id for group_id, block_id, _ in partial_tail_offloads
        }

        keys: list[str] = []
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        group_ids: list[str] | None = (
            [] if self.enable_group_semantics and self.supports_group_ids else None
        )
        saved = self._saved_offset.get(req_meta.req_id, 0)
        for g_idx, db in enumerate(self.token_databases):
            group_blocks = req_meta.block_ids[g_idx]
            # Distribute across ranks by the same rule as normal chunks.
            put_step = self.group_put_steps[g_idx]
            put_step_rank = (self.tp_rank + g_idx) % put_step
            # Always include the boundary block: its sub-hash key is written
            # only here, even if normal saves already advanced past it.
            last_block = cdiv(boundary, db.block_size) - 1
            for block_idx in range(
                min(saved // db.block_size, last_block), last_block + 1
            ):
                if block_idx % put_step != put_step_rank:
                    continue
                valid_end = min((block_idx + 1) * db.block_size, boundary)
                key_hash = req_meta.block_hashes[valid_end // hash_block_size - 1]
                if (
                    g_idx in mamba_offloads
                    and valid_end == boundary
                    and boundary % db.block_size != 0
                ):
                    block_id = mamba_offloads[g_idx]
                else:
                    if block_idx >= len(group_blocks):
                        continue
                    block_id = group_blocks[block_idx]
                if block_id == NULL_BLOCK_ID:
                    logger.debug(
                        "Skipping unavailable partial-tail source block "
                        "(req=%s, group=%d, block=%d)",
                        req_meta.req_id,
                        g_idx,
                        block_idx,
                    )
                    continue
                addr, size = db.prepare_value_for_block(block_id)
                key = db.key_for(key_hash)
                keys.append(key)
                addrs.append(addr)
                sizes.append(size)
                if group_ids is not None:
                    group_ids.append(
                        _make_mooncake_group_id(
                            db.metadata,
                            key.rsplit("@", 1)[-1],
                        )
                    )

        if not keys:
            return True
        exists_start = time.perf_counter()
        try:
            exists = self.store.batch_is_exist(keys)
        except Exception as e:
            self._record_operation(
                "save_exists",
                exists_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Failed to check partial-tail keys for request %s: %s",
                req_meta.req_id,
                e,
            )
            return False
        self._record_operation("save_exists", exists_start, len(keys))
        missing = [i for i, e in enumerate(exists) if e != 1]
        if not missing:
            return True
        keys = [keys[i] for i in missing]
        addrs = [addrs[i] for i in missing]
        sizes = [sizes[i] for i in missing]
        if group_ids is not None:
            group_ids = [group_ids[i] for i in missing]
        if req_meta.current_event is not None:
            # Fence the CoW block copy enqueued earlier this step.
            req_meta.current_event.synchronize()
        if group_ids is not None:
            assert len(group_ids) == len(keys)
            self.replicate_config.group_ids = group_ids
        batch_bytes = _sum_batch_bytes(sizes)
        put_start = time.perf_counter()
        try:
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, self.replicate_config
            )
        except Exception as e:
            self._record_operation(
                "save_put",
                put_start,
                len(keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Failed to put partial-tail keys for request %s: %s",
                req_meta.req_id,
                e,
            )
            return False

        failed = [i for i, value in enumerate(res) if value < 0]
        self._record_operation(
            "save_put",
            put_start,
            len(keys),
            num_bytes=batch_bytes,
            status="partial_failure" if failed else "ok",
            num_failed_keys=len(failed),
        )
        if failed:
            failed_codes = {res[i] for i in failed}
            logger.warning(
                "Partial-tail put failed for request %s: %d/%d keys failed (codes=%s)",
                req_meta.req_id,
                len(failed),
                len(keys),
                failed_codes,
            )
            if MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes:
                self._mark_request_skipped_for_pressure(req_meta)
            return False

        if self._clear_store_pressure():
            logger.info(
                "Mooncake CPU/disk offloading pressure cleared after a "
                "successful partial-tail batch"
            )
        return True

    def start_put_sessions(
        self,
        keys: list[str],
        object_size: int,
    ) -> None:
        """Start put sessions for save keys (one Master RPC per batch).

        Called by ``MooncakeStoreWorker._start_layerwise_sessions()`` before
        per-layer tasks are constructed.  Once a session is open, subsequent
        ``batch_put_from_multi_buffer_ranges()`` calls can write data at
        layer offsets without additional Master communication.

        The set of keys whose session actually opened is recorded in
        ``_active_put_keys`` so the per-layer range-put handler only writes
        live sessions and skips any whose start failed.
        """
        if not self._use_session_api:
            return
        if not keys:
            self._active_put_keys = set()
            return

        sizes = [object_size] * len(keys)
        try:
            results = self.store.batch_put_session_start(
                keys, sizes, self.replicate_config
            )
        except Exception as exc:
            logger.error("batch_put_session_start failed: %s", exc)
            self._revoke_range_keys(keys)
            self._active_put_keys = set()
            return

        failed = [k for k, r in zip(keys, results) if r != 0]
        if failed:
            logger.warning(
                "batch_put_session_start failed for keys: %s", failed
            )
            self._revoke_range_keys(failed)

        # Only keys whose session opened are writable this batch.
        self._active_put_keys = {k for k, r in zip(keys, results) if r == 0}

    def _revoke_range_keys(self, keys: list[str]) -> None:
        """Cancel unfinished put sessions (cleanup on failure)."""
        if not keys:
            return
        try:
            self.store.batch_put_session_revoke(keys)
        except Exception as exc:
            logger.error("batch_put_session_revoke failed: %s", exc)

    def _handle_layer_task(self, task: LayerTransferTask) -> None:
        """Legacy per-layer-key save path (Phase 1)."""
        keys = task.key_list
        addrs = task.addr_list
        sizes = task.size_list

        if not keys:
            self.request_queue.task_done()
            return

        req_id = task.req_id
        start_time = time.perf_counter()

        try:
            exist_mask = self.store.batch_is_exist(keys)

            actual_keys = [k for k, e in zip(keys, exist_mask) if not e]
            actual_addrs = [a for a, e in zip(addrs, exist_mask) if not e]
            actual_sizes = [s for s, e in zip(sizes, exist_mask) if not e]

            if actual_keys:
                batch_put_start = time.perf_counter()
                res = self.store.batch_put_from_multi_buffers(
                    actual_keys, actual_addrs, actual_sizes
                )

                self._record_operation(
                    "save_put_layer",
                    batch_put_start,
                    len(actual_keys),
                    num_bytes=sum(sum(s) for s in actual_sizes),
                    status="ok" if res else "error",
                )

                if not res:
                    logger.error(
                        "Layerwise save failed for layer %d, req %s",
                        task.physical_layer_id,
                        req_id,
                    )

            if self._layerwise_enabled:
                layer_id = task.physical_layer_id
                event = self._layer_save_finished_events.get(layer_id)
                if event:
                    event.set()

            if task.physical_layer_id == self._num_layers - 1:
                self.set_finished_request(req_id)

        except Exception as e:
            self._record_operation(
                "save_put_layer",
                start_time,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Layerwise save error for layer %d, req %s: %s",
                task.physical_layer_id,
                req_id,
                e,
            )
        finally:
            self.request_queue.task_done()

    def _handle_layer_range_task(self, task: LayerTransferTask) -> None:
        """Save a single layer via the Mooncake Session API.

        Session lifecycle:
        1. Layer 0: no extra work (session already open).
        2. Layer 1..N-2: ``batch_put_from_multi_buffer_ranges`` (zero Master RPC).
        3. Layer N-1 (last): ranges + ``batch_put_session_end`` to commit.
        On per-key failure the failed keys are revoked and excluded from
        subsequent layers.
        """
        keys = task.key_list
        layer_id = task.physical_layer_id

        if not keys:
            self.request_queue.task_done()
            return

        req_id = task.req_id
        start_time = time.perf_counter()

        try:
            # _active_put_keys is seeded by start_put_sessions() before the
            # forward with the keys whose session actually opened. If a step
            # reached the range-put handler without that (e.g. session start
            # was skipped), fall back to treating all keys as active.
            if self._active_put_keys is None:
                self._active_put_keys = set(keys)

            # Only write keys whose session is still alive
            active_indices = [
                i for i, k in enumerate(keys)
                if k in self._active_put_keys
            ]
            active_keys = [keys[i] for i in active_indices]

            if active_keys:
                results = self.store.batch_put_from_multi_buffer_ranges(
                    active_keys,
                    [task.addr_list[i] for i in active_indices],
                    [task.size_list[i] for i in active_indices],
                    [task.dst_offset_list[i] for i in active_indices],
                )

                self._record_operation(
                    "save_put_ranges",
                    start_time,
                    len(active_keys),
                    num_bytes=sum(
                        sum(task.size_list[i]) for i in active_indices
                    ),
                    status="ok",
                )

                # Revoke failed keys — they are dropped from subsequent layers
                failed_keys = [
                    k for k, r in zip(active_keys, results) if r < 0
                ]
                if failed_keys:
                    self._revoke_range_keys(failed_keys)
                    self._active_put_keys.difference_update(failed_keys)
                    logger.warning(
                        "Layer %d save: %d/%d range-put keys failed, req=%s",
                        layer_id,
                        len(failed_keys),
                        len(active_keys),
                        req_id,
                    )

            # Last layer: commit all surviving sessions
            if layer_id == self._num_layers - 1:
                commit_keys = [
                    k for k in keys if k in self._active_put_keys
                ]
                if commit_keys:
                    try:
                        commit_results = self.store.batch_put_session_end(
                            commit_keys
                        )
                        failed_commit = [
                            k
                            for k, r in zip(commit_keys, commit_results)
                            if r != 0
                        ]
                        if failed_commit:
                            self._revoke_range_keys(failed_commit)
                    except Exception as exc:
                        logger.error(
                            "batch_put_session_end failed: %s", exc
                        )
                        self._revoke_range_keys(commit_keys)

                self._active_put_keys = None
                self.set_finished_request(req_id)

            # Signal layer completion
            if self._layerwise_enabled:
                event = self._layer_save_finished_events.get(layer_id)
                if event:
                    event.set()

        except Exception as e:
            self._record_operation(
                "save_put_ranges",
                start_time,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Session save error for layer %d, req %s: %s",
                layer_id,
                req_id,
                e,
            )
            # On catastrophic failure, revoke all keys for this batch
            self._revoke_range_keys(keys)
            self._active_put_keys = None
        finally:
            self.request_queue.task_done()

    def _handle_request(self, req_meta: ReqMeta | LayerTransferTask):
        # ============================================================
        # Layerwise support: dispatch to layer-specific handler
        # ============================================================
        if isinstance(req_meta, LayerTransferTask):
            if req_meta.use_key_major_ranges and self._use_session_api:
                self._handle_layer_range_task(req_meta)
            else:
                self._handle_layer_task(req_meta)
            return

        # The single `finally` is the only way out, so the scheduler releases
        # this job's GPU block references however the job ends.
        try:
            # Cache hits are always a multiple of ``lcm_block_size`` tokens,
            # which is also ``store_mask``'s precondition.
            lcm_block_size = self.coord.lcm_block_size
            token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
            block_ids_per_group = req_meta.block_ids
            req_id = req_meta.req_id
            current_event = req_meta.current_event

            if not self.is_live_store_job(req_meta):
                return

            if self._should_skip_request(req_id):
                logger.debug(
                    "Skipping Mooncake store for request %s while CPU/disk "
                    "offloading is under pressure",
                    req_id,
                )
                return

            # Offload the sub-block partial tail (independent of the normal
            # block-aligned save, which may be skipped this step).
            if req_meta.partial_tail_offloads is not None and not (
                self._maybe_offload_partial_tail(req_meta)
            ):
                return

            if token_len == 0:
                return

            # Resume from where this rank left off; only the new suffix is saved.
            save_start = self._saved_offset.get(req_id, 0)

            # Within each lcm region only per-spec relevant chunks are loaded
            # (e.g., SWA or linear attn), so mask out irrelevant chunks
            store_masks = self.coord.store_mask(
                token_len,
                save_start,
                num_prompt_tokens=req_meta.num_prompt_tokens,
            )

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            kv_event_block_hashes: list[BlockHash] = []
            group_indices: list[int] = []
            for g_idx, db in enumerate(self.token_databases):
                # Rotate the stride phase per group to balance load across ranks.
                put_step = self.group_put_steps[g_idx]
                put_step_rank = (self.tp_rank + g_idx) % put_step
                for start, end, block_hash in db.process_tokens(
                    token_len,
                    req_meta.block_hashes,
                    mask_num=save_start,
                    chunk_mask=store_masks[g_idx],
                    put_step=put_step,
                    put_step_rank=put_step_rank,
                ):
                    starts.append(start)
                    ends.append(end)
                    keys.append(db.key_for(block_hash))
                    if self.enable_kv_event:
                        kv_event_block_hashes.append(block_hash)
                    group_indices.append(g_idx)

            if not keys:
                self._record_saved(req_meta, token_len)
                return

            # Check which blocks already exist (dedup)
            save_exists_start = time.perf_counter()
            try:
                exists_states = self.store.batch_is_exist(keys)
            except Exception:
                self._record_operation(
                    "save_exists",
                    save_exists_start,
                    len(keys),
                    status="error",
                    num_failed_keys=len(keys),
                )
                raise
            self._record_operation(
                "save_exists",
                save_exists_start,
                len(keys),
            )
            missing_indices = [
                i for i, exists in enumerate(exists_states) if exists != 1
            ]

            if not missing_indices:
                self._record_saved(req_meta, token_len)
                return

            if len(missing_indices) != len(keys):
                starts = [starts[i] for i in missing_indices]
                ends = [ends[i] for i in missing_indices]
                keys = [keys[i] for i in missing_indices]
                if self.enable_kv_event:
                    kv_event_block_hashes = [
                        kv_event_block_hashes[i] for i in missing_indices
                    ]
                group_indices = [group_indices[i] for i in missing_indices]

            group_ids = (
                [
                    _make_mooncake_group_id(
                        self.token_databases[g_idx].metadata,
                        key.rsplit("@", 1)[-1],
                    )
                    for key, g_idx in zip(keys, group_indices, strict=True)
                ]
                if self.enable_group_semantics and self.supports_group_ids
                else None
            )

            logger.debug(
                "Storing KV cache for %d blocks (groups=%s) for request %s",
                len(keys),
                set(group_indices),
                req_id,
            )

            addrs: list[list[int]] = []
            sizes: list[list[int]] = []
            stored_events: list[BlockStored] = []
            chunks_per_group: list[list[tuple[int, int]]] = [
                [] for _ in self.token_databases
            ]
            for start, end, g_idx in zip(starts, ends, group_indices, strict=True):
                chunks_per_group[g_idx].append((start, end))
            for g_idx, chunks in enumerate(chunks_per_group):
                if not chunks:
                    continue
                db = self.token_databases[g_idx]
                group_addrs, group_sizes, _ = db.prepare_values(
                    chunks, block_ids_per_group[g_idx]
                )
                addrs.extend(group_addrs)
                sizes.extend(group_sizes)

            # parent_block_hash chains live within a group, not across.
            if self.enable_kv_event:
                prev_key_per_group: dict[int, Any] = {}
                new_block_hashes = [
                    maybe_convert_block_hash(bh) for bh in kv_event_block_hashes
                ]

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]
                if self.enable_kv_event:
                    token_ids = (
                        req_meta.token_ids[s:e]
                        if req_meta.token_ids is not None
                        else None
                    )
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[idx]],
                        parent_block_hash=prev_key_per_group.get(g_idx),
                        token_ids=token_ids,
                        block_size=db.block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                        group_idx=g_idx,
                    )
                    stored_events.append(stored_event)
                    prev_key_per_group[g_idx] = new_block_hashes[idx]

            if current_event is not None:
                current_event.synchronize()

            if group_ids is not None:
                assert len(group_ids) == len(keys)
                self.replicate_config.group_ids = group_ids

            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
                failed = [i for i, v in enumerate(res) if v < 0]
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    failed_codes = set(res[i] for i in failed)
                    logger.warning(
                        "batch_put failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d, num_keys=%d), "
                        "first_key=%s",
                        len(failed),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        len(keys),
                        keys[0] if keys else "N/A",
                    )
                    if (
                        MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                        and not self._mark_request_skipped_for_pressure(req_meta)
                    ):
                        logger.warning(
                            "Detected Mooncake CPU/disk offloading pressure "
                            "(NO_AVAILABLE_HANDLE); skipping future store "
                            "batches for request %s until a later store "
                            "batch succeeds",
                            req_id,
                        )
                else:
                    self._record_saved(req_meta, token_len)
                    if self._clear_store_pressure():
                        logger.info(
                            "Mooncake CPU/disk offloading pressure cleared "
                            "after a successful store batch"
                        )
            except Exception as e:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(keys),
                )
                logger.error("Failed to put key %s, error: %s", keys, e)

            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)
        finally:
            self.finish_store_job(req_meta)
            self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
            request_queue=request_queue,
        )
        # _invalid_block_ids can be access by both the Worker and RecvingThread
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )
        self.coord = coord

        # Session API state
        self._active_load_indices: set[int] | None = None

    def start_get_sessions(self, keys: list[str]) -> list[str]:
        """Start get sessions for load keys (one Master RPC).

        Called by ``MooncakeStoreWorker._start_layerwise_sessions()``.

        Returns the keys whose session actually opened. Fails gracefully on
        partial failure: errored sessions are revoked and excluded from later
        layers so the surviving sessions remain usable.
        """
        if not self._use_session_api:
            return []
        if not keys:
            return []
        try:
            results = self.store.batch_get_session_start(keys)
        except Exception as exc:
            logger.error("batch_get_session_start failed: %s", exc)
            try:
                self.store.batch_get_session_end(keys)
            except Exception:
                pass
            return []
        failed = [k for k, r in zip(keys, results) if r != 0]
        opened = [k for k, r in zip(keys, results) if r == 0]
        if failed:
            logger.warning(
                "batch_get_session_start missed/errored keys: %s", failed
            )
            # Revoke sessions that failed to open so a fresh start_get_sessions
            # on the next step cannot collide with a stale partial lease.
            try:
                self.store.batch_get_session_end(failed)
            except Exception:
                pass

        return opened

    def end_get_sessions(self, keys: list[str]) -> None:
        """Release get sessions (one shot per step).

        Called by ``MooncakeStoreWorker._close_load_sessions_once()``.
        """
        if not self._use_session_api or not keys:
            return
        try:
            self.store.batch_get_session_end(keys)
        except Exception as exc:
            logger.error("batch_get_session_end failed: %s", exc)

    def _add_load_error_block_ids(self, block_ids: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def _handle_layer_task(self, task: LayerTransferTask) -> None:
        """Legacy per-layer-key load path (Phase 1)."""
        keys = task.key_list
        addrs = task.addr_list
        sizes = task.size_list
        block_ids = task.block_ids

        if not keys:
            self.request_queue.task_done()
            return

        req_id = task.req_id
        layer_id = task.physical_layer_id

        try:
            load_start = time.perf_counter()
            results = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)

            # Detect blocks that failed to load.
            failed = [
                (key, value, block_id)
                for key, value, block_id in zip(keys, results, block_ids, strict=True)
                if value < 0
            ]

            self._record_operation(
                "load_get_layer",
                load_start,
                len(keys),
                status="partial_failure" if failed else "ok",
                num_failed_keys=len(failed),
            )

            if failed:
                self._add_load_error_block_ids([block_id for _, _, block_id in failed])
                logger.warning(
                    "Layerwise load failed for layer %d, req %s: %d keys failed",
                    layer_id,
                    req_id,
                    len(failed),
                )

            if self._layerwise_enabled:
                event = self._layer_load_finished_events.get(layer_id)
                if event:
                    event.set()

            if layer_id == self._num_layers - 1:
                self.set_finished_request(req_id)

        except Exception as e:
            self._record_operation(
                "load_get_layer",
                time.perf_counter(),
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Layerwise load error for layer %d, req %s: %s",
                layer_id,
                req_id,
                e,
            )
        finally:
            self.request_queue.task_done()

    def _handle_layer_range_task(self, task: LayerTransferTask) -> None:
        """Load a single layer via the Mooncake Session API.

        Session lifecycle:
        1. Layer 0: init active-index set (all keys start active).
        2. Layer 1..N-2: ``batch_get_into_multi_buffer_ranges`` (zero Master RPC).
        3. Layer N-1 (last): ranges + finalize request tracking.
        On per-key failure the failed indices are marked as invalid and
        excluded from subsequent layers.
        """
        keys = task.key_list
        layer_id = task.physical_layer_id

        if not keys:
            self.request_queue.task_done()
            return

        req_id = task.req_id

        try:
            # Init active-index set on the first layer
            if self._active_load_indices is None or layer_id == 0:
                self._active_load_indices = set(range(len(keys)))

            # Filter: only load indices whose get session is still alive
            active_indices = [
                i
                for i in range(len(keys))
                if i in self._active_load_indices
            ]
            active_keys = [keys[i] for i in active_indices]

            if active_keys:
                load_start = time.perf_counter()
                results = self.store.batch_get_into_multi_buffer_ranges(
                    active_keys,
                    [task.addr_list[i] for i in active_indices],
                    [task.size_list[i] for i in active_indices],
                    [task.dst_offset_list[i] for i in active_indices],
                )

                self._record_operation(
                    "load_get_ranges",
                    load_start,
                    len(active_keys),
                    status="ok",
                )

                # Mark failed indices and drop them from subsequent layers
                failed_indices = [
                    i
                    for i, r in zip(active_indices, results)
                    if r < 0
                ]
                if failed_indices:
                    self._add_load_error_block_ids(
                        [task.block_ids[i] for i in failed_indices]
                    )
                    self._active_load_indices.difference_update(failed_indices)
                    logger.warning(
                        "Layer %d load: %d/%d range-get keys failed, req=%s",
                        layer_id,
                        len(failed_indices),
                        len(active_keys),
                        req_id,
                    )

            # Last layer: finalize the request
            if layer_id == self._num_layers - 1:
                self.set_finished_request(req_id)
                self._active_load_indices = None

            # Signal layer completion
            if self._layerwise_enabled:
                event = self._layer_load_finished_events.get(layer_id)
                if event:
                    event.set()

        except Exception as e:
            self._record_operation(
                "load_get_ranges",
                time.perf_counter(),
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Session load error for layer %d, req %s: %s",
                layer_id,
                req_id,
                e,
            )
            self._active_load_indices = None
        finally:
            self.request_queue.task_done()

    def _handle_request(self, req_meta: ReqMeta | LayerTransferTask):
        # ============================================================
        # Layerwise support: dispatch to layer-specific handler
        # ============================================================
        if isinstance(req_meta, LayerTransferTask):
            if req_meta.use_key_major_ranges:
                self._handle_layer_range_task(req_meta)
            else:
                self._handle_layer_task(req_meta)
            return

        # Original non-layerwise handling (ReqMeta)
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )

        # Skip chunks the consumer's per-group spec wouldn't populate
        # locally (e.g. SWA pre-window) even if the producer stored them.
        load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)

        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        for g_idx, db in enumerate(self.token_databases):
            mask = load_mask_per_group[g_idx]
            chunks: list[tuple[int, int]] = []
            for start, end, block_hash in db.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                key_list.append(db.key_for(block_hash))
                chunks.append((start, end))
            g_addrs, g_sizes, g_block_ids = db.prepare_values(
                chunks, req_meta.block_ids[g_idx]
            )
            addr_list.extend(g_addrs)
            size_list.extend(g_sizes)
            block_id_list.extend(g_block_ids)

        # Rotate aligned lists by tp_rank for load balancing.
        rotation = self.tp_rank % len(key_list)
        key_list_c = _rotate_list(key_list, rotation)
        addr_list_c = _rotate_list(addr_list, rotation)
        size_list_c = _rotate_list(size_list, rotation)
        block_id_list_c = _rotate_list(block_id_list, rotation)

        load_batches = [(key_list_c, addr_list_c, size_list_c, block_id_list_c)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in size_list_c
            )
            if total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes:
                assert self.disk_offload_buffer_budget_bytes is not None
                split_batches, oversized_key = _split_disk_offload_load_batches(
                    key_list_c,
                    addr_list_c,
                    size_list_c,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    oversized_key_index = key_list_c.index(oversized_key)
                    # Mark every block: we skip the whole request, and the
                    # tp_rank rotation means oversized_key isn't necessarily
                    # the first block in the request's original order.
                    self._add_load_error_block_ids(block_id_list_c)
                    oversized_key_bytes = _estimate_disk_offload_staging_bytes(
                        size_list_c[oversized_key_index]
                    )
                    logger.warning(
                        "Skipping Mooncake load for request %s because key %s "
                        "requires %d staging bytes, exceeding budget %d",
                        req_id,
                        oversized_key,
                        oversized_key_bytes,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    self.set_finished_request(req_id)
                    self.request_queue.task_done()
                    return
                load_batches = []
                block_id_offset = 0
                for batch_keys, batch_addrs, batch_sizes in split_batches:
                    next_block_id_offset = block_id_offset + len(batch_keys)
                    batch_block_ids = block_id_list_c[
                        block_id_offset:next_block_id_offset
                    ]
                    load_batches.append(
                        (batch_keys, batch_addrs, batch_sizes, batch_block_ids)
                    )
                    block_id_offset = next_block_id_offset

        current_batch_keys: list[str] = key_list_c
        current_batch_block_ids: list[int] = block_id_list_c
        batch_bytes = 0
        try:
            for batch_keys, batch_addrs, batch_sizes, batch_block_ids in load_batches:
                current_batch_keys = batch_keys
                current_batch_block_ids = batch_block_ids
                batch_bytes = _sum_batch_bytes(batch_sizes)
                tiers_by_key: dict[str, str] | None = None
                if envs.VLLM_MOONCAKE_STORE_TIER_LOG:
                    tiers_by_key = _get_replica_tiers_by_key(self.store, batch_keys)
                # Reset so the recorded RPC duration excludes tier lookup.
                load_get_start = time.perf_counter()
                res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )
                if tiers_by_key is not None:
                    _log_mooncake_load_tier_summary(
                        req_id, batch_keys, res, tiers_by_key
                    )
                failed = [
                    (key, value, block_id)
                    for key, value, block_id in zip(
                        batch_keys, res, batch_block_ids, strict=True
                    )
                    if value < 0
                ]
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(batch_keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    self._add_load_error_block_ids(
                        [block_id for _, _, block_id in failed]
                    )
                    logger.warning(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        [(key, value) for key, value, _ in failed[:3]],
                    )
                    break
        except Exception as e:
            self._add_load_error_block_ids(current_batch_block_ids)
            self._record_operation(
                "load_get",
                load_get_start,
                len(current_batch_keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(current_batch_keys),
            )
            logger.warning(
                "Failed to get Mooncake sub-batch %s, error: %s",
                current_batch_keys[:3],
                e,
            )

        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
# Store Worker
# ============================================================


class MooncakeStoreWorker:

    _use_session_api: bool = False
    """Worker-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        try:
            from mooncake.store import (  # type: ignore
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to run vLLM with MooncakeStoreConnector."
            ) from e

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.dp_rank = parallel_config.data_parallel_index
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", True
        )
        # Mirrors MooncakeStoreConnector._capacity_only.
        self._capacity_only = self.kv_role == "kv_consumer" and not (
            vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "enable_lookup", True
            )
        )
        self.cache_config = vllm_config.cache_config
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.num_layers = model_config.get_num_layers(parallel_config)

        self.num_kv_head = model_config.get_total_num_kv_heads()

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
        extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
            if vllm_config.kv_transfer_config
            else {}
        )
        self.store = MooncakeDistributedStore()
        local_ip = get_ip()
        local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
        setup_kwargs: dict[str, str] = {}
        if store_config.tenant_id != DEFAULT_TENANT_ID:
            setup_kwargs["tenant_id"] = store_config.tenant_id
        ret = self.store.setup(
            local_hostname,
            store_config.metadata_server,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
            **setup_kwargs,
        )
        if ret != 0:
            msg = "Initialize MooncakeDistributedStore failed."
            logger.error(msg)
            raise RuntimeError(msg)

        preferred_segment = rdma_utils.get_configured_preferred_segment(extra_config)
        self.preferred_segment = preferred_segment
        self.store_replicate_config = ReplicateConfig()
        self.enable_group_semantics = (
            str(extra_config.get("enable_group_semantics", "False")).strip().lower()
            == "true"
        )
        self._supports_group_ids = _replicate_config_supports_group_ids(
            ReplicateConfig, self.store_replicate_config
        )
        if self.enable_group_semantics and not self._supports_group_ids:
            logger.warning(
                "Mooncake group semantics is enabled, but the installed "
                "Mooncake package does not support ReplicateConfig.group_ids. "
                "Falling back to the existing batch_put_from_multi_buffers path."
            )
        if preferred_segment is not None:
            self.store_replicate_config.preferred_segment = preferred_segment

        logger.info(
            "Mooncake mode=%s (global_segment_size=%d, local_buffer_size=%d, "
            "preferred_segment=%s, enable_offload=%s, tenant_id=%s)",
            store_config.mode,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            preferred_segment or "<none>",
            store_config.enable_offload,
            store_config.tenant_id,
        )
        if store_config.mode == "embedded":
            if store_config.enable_offload and preferred_segment is None:
                logger.warning(
                    "enable_offload is set in embedded mode without "
                    "preferred_segment; SSD tier will only see puts that "
                    "happen to land on the owner segment."
                )
            if preferred_segment is not None:
                logger.warning(
                    "preferred_segment=%s with mode=embedded: rank-"
                    "contributed segments will be idle.",
                    preferred_segment,
                )
        elif (
            store_config.mode == "standalone-store" and not store_config.enable_offload
        ):
            logger.warning(
                "standalone-store mode without enable_offload: large prefills "
                "may exceed the owner DirectIO budget."
            )

        self.disk_offload_buffer_budget_bytes = (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES
            if store_config.enable_offload
            else None
        )

        # Start lookup server on rank 0 for scheduler-side prefix queries
        self.lookup_server: LookupKeyServer | None = None
        if vllm_config.parallel_config.rank == 0:
            self.lookup_server = LookupKeyServer(self, vllm_config)

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        # Pool of load-receive threads
        self.kv_recv_threads: list[KVCacheStoreRecvingThread] = []
        self.num_recv_threads = max(1, envs.VLLM_MOONCAKE_LOAD_RECV_THREADS)
        self.recv_request_queue: queue.Queue[ReqMeta] = queue.Queue()
        self.finished_store_req: set[str] = set()
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()

        self._kv_cache_config = kv_cache_config
        self.token_dbs: list[ChunkedTokenDatabase] = []

        # a capacity-only instance does not need below utils
        if self._capacity_only:
            logger.info(
                "Mooncake store in capacity-only mode: segment mounted "
                "(global_segment_size=%d), KV transfer disabled.",
                store_config.global_segment_size,
            )
            return

        # Single-group + PCP/DCP > 1: scale the lone group's spec.block_size to
        # self.block_size (= scheduler_block_size) so the coordinator's
        # ``block_size % hash_block_size == 0`` invariant holds.
        groups = list(kv_cache_config.kv_cache_groups)
        if len(groups) == 1 and groups[0].kv_cache_spec.block_size != self.block_size:
            g = groups[0]
            groups = [
                dataclasses.replace(
                    g,
                    kv_cache_spec=dataclasses.replace(
                        g.kv_cache_spec, block_size=self.block_size
                    ),
                )
            ]
        self._kv_cache_groups: list[KVCacheGroupSpec] = groups
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            spec_cfg.use_eagle()
            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
            else False
        )
        self.coord = MooncakeStoreCoordinator(
            self._kv_cache_groups,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle,
            retention_interval=kv_cache_config.prefix_cache_retention_interval,
        )
        # One ChunkedTokenDatabase per group; addresses populated in
        # register_kv_caches once the kv-cache layout is known. Each group's
        # key namespace is its TP shard id: ranks holding identical bytes
        # (MLA / shared GQA KV heads) share a namespace, TP-sharded Mamba
        # state gets one namespace per rank.
        metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            tp_rank=self.tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
            cache_prefix=str(
                vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                    "cache_prefix", ""
                )
            ),
        )
        self._group_tp_replication_factors: tuple[int, ...] = (
            self._compute_group_tp_replication_factors()
        )
        self.token_dbs = [
            ChunkedTokenDatabase(
                dataclasses.replace(
                    metadata,
                    group_id=g_idx,
                    tp_rank=self.tp_rank // self._group_tp_replication_factors[g_idx],
                ),
                g.kv_cache_spec.block_size,
                hash_block_size=self.hash_block_size,
            )
            for g_idx, g in enumerate(self._kv_cache_groups)
        ]
        self._init_lookup_key_prefixes()

        # ============================================================
        # Layerwise support
        # ============================================================
        self._layerwise_enabled = False
        self._use_session_api = _mooncake_supports_session_api(self.store)
        self._layer_save_tasks: dict[int, list[LayerTransferTask]] = {}
        self._layer_load_tasks: dict[int, list[LayerTransferTask]] = {}
        self._layer_save_finished_events: dict[int, threading.Event] = {}
        self._layer_load_finished_events: dict[int, threading.Event] = {}
        self._current_save_layer: int = 0
        self._current_load_layer: int = 0
        self._next_load_layer_to_submit: int = 0
        self._num_prefetch_layers: int = 1
        self._save_finalized: bool = False

        # Read layerwise parameters from the extra config.
        kvc_extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        if kvc_extra:
            self._layerwise_enabled = str(kvc_extra.get("use_layerwise", "False")).lower() == "true"
            self._num_prefetch_layers = int(kvc_extra.get("layerwise_prefetch_layers", 2))

        # Layerwise mode embeds KV load into the model forward pass.
        # load_async (which reports finished_recving to the scheduler,
        # expecting requests in WAITING_FOR_REMOTE_KVS) is incompatible
        # with layerwise mode where requests enter RUNNING directly.
        if self._layerwise_enabled:
            self.load_async = False
            if self._use_session_api:
                logger.info(
                    "Mooncake session API available, using ranged "
                    "multi-buffer transfer for layerwise mode"
                )
            else:
                logger.warning(
                    "Mooncake session API not available, falling back "
                    "to per-layer-key transfer for layerwise mode"
                )
            self._init_layerwise_config()

    def _spec_tp_replication_factor(self, spec: KVCacheSpec) -> int:
        if self.dcp_size > 1:
            return 1
        inner_specs = (
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        # Any rank-specific state makes the whole packed value rank-specific.
        if any(isinstance(inner, MambaSpec) for inner in inner_specs):
            return 1
        # A pure MLA packed value is replicated on every TP rank.
        if all(
            isinstance(inner, (MLAAttentionSpec, SlidingWindowMLASpec))
            for inner in inner_specs
        ):
            return self.tp_size
        return max(1, self.tp_size // self.num_kv_head)

    def _init_layerwise_config(self) -> None:
        """Initialize per-layer state (events, task lists, session API)."""
        if not self._kv_cache_groups:
            logger.warning("No KV cache groups configured for layerwise mode")
            return

        self._num_layers = self.num_layers  # Already set from model_config

        self._layer_save_finished_events = {}
        self._layer_load_finished_events = {}
        self._layer_save_tasks = {}
        self._layer_load_tasks = {}

        for layer_id in range(self._num_layers):
            self._layer_save_finished_events[layer_id] = threading.Event()
            self._layer_load_finished_events[layer_id] = threading.Event()
            self._layer_save_tasks[layer_id] = []
            self._layer_load_tasks[layer_id] = []

        if self._use_session_api:
            self._load_sessions_closed = False
            self._load_session_lock = threading.Lock()
            self._opened_load_keys: list[str] = []
            # Placeholder: block_len is empty at __init__ time (it is filled
            # later in register_kv_caches), so _compute_page_size_bytes() would
            # return 0 here. The real value is (re)computed in register_kv_caches
            # once the kv-cache layout is known.
            self._page_size_bytes = 0

        logger.info(
            "Layerwise mode enabled: num_layers=%d, prefetch_layers=%d%s",
            self._num_layers,
            self._num_prefetch_layers,
            ", session_api=True" if self._use_session_api else "",
        )

    def _compute_page_size_bytes(self) -> int:
        """Compute the per-layer page size in bytes for Session API offset.

        Uses the first group's ChunkedTokenDatabase.  When num_layers is set,
        block_len is sliced by ``caches_per_layer = len(block_len) // num_layers``.
        """
        if not self.token_dbs:
            return 0
        db = self.token_dbs[0]
        caches_per_layer = (
            len(db.block_len) // max(1, db.num_layers) if db.num_layers > 0
            else len(db.block_len)
        )
        if caches_per_layer == 0:
            caches_per_layer = len(db.block_len)
        return sum(db.block_len[:caches_per_layer])

    def _start_layerwise_sessions(self, requests: list[ReqMeta]) -> None:
        """Start put/get sessions before per-layer tasks are built.

        Called from ``start_load_kv()`` *before* ``_build_layer_tasks_from_requests()``
        so that subsequent per-layer ``batch_put_from_multi_buffer_ranges`` /
        ``batch_get_into_multi_buffer_ranges`` calls operate on open sessions (zero
        additional Master RPCs).

        Put session:
          Reserve object space on the store side via ``batch_put_session_start``.
          The object size is ``page_size_bytes × num_layers`` — one linear region
          with layer i at byte offset ``i × page_size_bytes``.

        Get session:
          Query replica location + acquire lease via ``batch_get_session_start``.
        """
        if not self._use_session_api:
            return

        object_size = self._page_size_bytes * self._num_layers

        # ---- Collect save keys (deduplicated across requests) ----
        save_keys: list[str] = []
        save_keys_set: set[str] = set()
        for req_meta in requests:
            if not req_meta.can_save:
                continue
            for group_id in range(len(req_meta.block_ids)):
                db = self.token_dbs[group_id]
                put_step = self._group_tp_replication_factors[group_id]
                put_step_rank = (self.tp_rank + group_id) % put_step
                for _, _, block_hash in db.process_tokens(
                    req_meta.token_len_chunk,
                    req_meta.block_hashes,
                    put_step=put_step,
                    put_step_rank=put_step_rank,
                ):
                    key = db.key_for(block_hash)
                    if key not in save_keys_set:
                        save_keys.append(key)
                        save_keys_set.add(key)

        if save_keys and self.kv_send_thread is not None:
            self.kv_send_thread.start_put_sessions(save_keys, object_size)
        elif self.kv_send_thread is not None:
            # No session phase this step — clear any stale active-key set so
            # the range-put handler cannot reuse keys from a previous step.
            self.kv_send_thread._active_put_keys = set()

        # ---- Collect load keys (deduplicated across requests) ----
        load_keys: list[str] = []
        load_keys_set: set[str] = set()
        for req_meta in requests:
            load_spec = req_meta.load_spec
            if load_spec is None or not load_spec.can_load:
                continue
            mask_num = (
                load_spec.vllm_cached_tokens
                // self.block_size * self.block_size
            )
            for group_id in range(len(req_meta.block_ids)):
                db = self.token_dbs[group_id]
                for _, _, block_hash in db.process_tokens(
                    load_spec.token_len,
                    req_meta.block_hashes,
                    mask_num,
                ):
                    key = db.key_for(block_hash)
                    if key not in load_keys_set:
                        load_keys.append(key)
                        load_keys_set.add(key)

        if load_keys:
            for recv_thread in self.kv_recv_threads:
                recv_thread.start_get_sessions(load_keys)
            with self._load_session_lock:
                self._opened_load_keys = load_keys
                self._load_sessions_closed = False

    def _close_load_sessions_once(self) -> None:
        """Release all get sessions (one-shot per step).

        Called at the end of the final layer's ``save_kv_layer``, after
        ``_wait_for_all_layer_saves()`` and before ``_reset_layer_state()``.
        At this point every layer has been loaded and saved, so the lease
        can be released safely.
        """
        if not self._use_session_api:
            return
        with self._load_session_lock:
            if self._load_sessions_closed:
                return
            self._load_sessions_closed = True
            keys = list(self._opened_load_keys)

        if keys:
            for recv_thread in self.kv_recv_threads:
                recv_thread.end_get_sessions(keys)

    def _compute_group_tp_replication_factors(self) -> tuple[int, ...]:
        """Return the number of byte-identical TP replicas per cache group.

        DCP and Mamba use 1; MLA uses ``tp_size``; GQA uses
        ``tp_size // num_kv_head``.
        """
        return tuple(
            self._spec_tp_replication_factor(group.kv_cache_spec)
            for group in self._kv_cache_groups
        )

    def _init_lookup_key_prefixes(self) -> None:
        def rank_namespaces(factor: int) -> tuple[tuple[int, int, int, int], ...]:
            if self.dcp_size > 1:
                # DCP is a TP subdivision: dcp_rank == tp_rank % dcp_size.
                return tuple(
                    (tp_rank, pcp_rank, tp_rank % self.dcp_size, pp_rank)
                    for pcp_rank in range(self.pcp_size)
                    for tp_rank in range(self.tp_size)
                    for pp_rank in range(self.pp_size)
                )
            return tuple(
                (shard_rank, pcp_rank, 0, pp_rank)
                for pcp_rank in range(self.pcp_size)
                for shard_rank in range(self.tp_size // factor)
                for pp_rank in range(self.pp_size)
            )

        self._lookup_key_prefixes = tuple(
            tuple(
                PoolKey.build_prefix(
                    db.metadata,
                    tp_rank=tp_rank,
                    pcp_rank=pcp_rank,
                    dcp_rank=dcp_rank,
                    pp_rank=pp_rank,
                )
                for tp_rank, pcp_rank, dcp_rank, pp_rank in rank_namespaces(
                    self._group_tp_replication_factors[g_idx]
                )
            )
            for g_idx, db in enumerate(self.token_dbs)
        )
    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor.

        Wraps the unified tensor in a single-entry dict so that the
        existing stride-based logic in register_kv_caches() produces
        the correct single-segment result (block_len = page_size * num_layers).
        """
        self.register_kv_caches({"__cross_layer__": kv_cache})

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register KV cache tensors and start transfer threads."""
        if self._capacity_only:
            return
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        seen_ptrs: set[int] = set()
        addrs: list[int] = []
        block_lens: list[int] = []

        for value in kv_caches.values():
            cache = _repr_tensor(value)
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            if base_addr in seen_ptrs:
                continue
            seen_ptrs.add(base_addr)
            region_len = cache_storage.nbytes()

            ret = self.store.register_buffer(base_addr, region_len)
            if ret != 0:
                logger.error(
                    "register_buffer failed for addr %#x len %d: %d",
                    base_addr,
                    region_len,
                    ret,
                )

            # Detect layout via stride: a dim whose byte-stride exceeds
            # page_size_bytes is an outer segment dim (e.g. the K/V dim of
            # FlashAttn's (2, num_blocks, ...)). FlashInfer/MLA's blocks-
            # outermost layout has no such dim and yields a single segment.
            el = cache.element_size()
            page_size_bytes = region_len // self.num_blocks
            outer_dims = [
                d for d in range(cache.ndim) if cache.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                # Blocks-first layout (FlashInfer / MLA): one segment.
                addrs.append(base_addr)
                block_lens.append(page_size_bytes)
            else:
                # K/V-first layout (FlashAttn / ROCm): split segments.
                seg_stride = cache.stride(outer_dims[0]) * el
                for idx in range(cache.shape[outer_dims[0]]):
                    addrs.append(base_addr + idx * seg_stride)
                    block_lens.append(seg_stride // self.num_blocks)

        logger.info(
            "Registered KV caches: num_groups=%d, num_segments=%d, num_blocks=%d",
            len(self.token_dbs),
            len(addrs),
            self.num_blocks,
        )

        for db in self.token_dbs:
            db.set_kv_caches_base_addr(addrs)
            db.set_block_len(block_lens)

        # ============================================================
        # Layerwise: set num_layers on each ChunkedTokenDatabase so
        # prepare_values_for_layer() can extract the target layer's
        # segment via block_len slicing.
        # ============================================================
        if self._layerwise_enabled and addrs and block_lens:
            for db in self.token_dbs:
                db.num_layers = self.num_layers
            logger.info(
                "Layerwise mode enabled with %d layers, %d segments, "
                "using per-layer slicing for address calculation",
                self.num_layers,
                len(addrs),
            )
            # block_len is only populated here, so the Session-API page size
            # cannot be computed in __init__. Recompute it now that the kv-cache
            # layout is known; otherwise object_size stays 0 and every
            # batch_put_session_start fails (rc=-600).
            if self._use_session_api:
                self._page_size_bytes = self._compute_page_size_bytes()

        # Start transfer threads
        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                self._group_tp_replication_factors,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
                self.store_replicate_config,
                enable_group_semantics=self.enable_group_semantics,
                supports_group_ids=self._supports_group_ids,
                record_operation=self._record_kv_connector_operation,
            )
            # Enable layerwise mode in sending thread
            if self._layerwise_enabled:
                self.kv_send_thread.enable_layerwise(self._num_layers)
                self.kv_send_thread._use_session_api = getattr(
                    self, '_use_session_api', False
                )
                for layer_id in range(self._num_layers):
                    self.kv_send_thread.set_layer_finished_event(
                        layer_id, True, self._layer_save_finished_events[layer_id]
                    )
            self.kv_send_thread.start()

        self.kv_recv_threads = []
        ready_events_recving = []
        for i in range(self.num_recv_threads):
            ready_event_recving = threading.Event()
            recv_thread = KVCacheStoreRecvingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                ready_event_recving,
                disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
                record_operation=self._record_kv_connector_operation,
                request_queue=self.recv_request_queue,
            )
            # Enable layerwise mode in receiving thread
            if self._layerwise_enabled:
                recv_thread.enable_layerwise(self._num_layers)
                recv_thread._use_session_api = self._use_session_api
                for layer_id in range(self._num_layers):
                    recv_thread.set_layer_finished_event(
                        layer_id, False, self._layer_load_finished_events[layer_id]
                    )
            recv_thread.name = f"KVCacheStoreRecvingThread-{i}"
            recv_thread.start()
            self.kv_recv_threads.append(recv_thread)
            ready_events_recving.append(ready_event_recving)
        for ready_event_recving in ready_events_recving:
            ready_event_recving.wait()
        logger.info(
            "Started %d Mooncake KV-load receive thread(s)", self.num_recv_threads
        )

    def start_load_kv(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """Build per-layer tasks before model forward.

        In layerwise mode, per-layer save/load tasks must be built
        here (before the model forward pass) so that
        wait_for_layer_load() and save_kv_layer() (called during the
        forward pass) have tasks to consume.

        When the Session API is available, put/get sessions
        are started here *before* task construction so that subsequent
        per-layer _ranges calls operate on open sessions.
        """
        if self._layerwise_enabled:
            # Backfill load_spec.token_len before starting sessions. It defaults
            # to 0 and _start_layerwise_sessions() reads it to decide which
            # blocks to lease via batch_get_session_start(); leaving it empty
            # means no get session is opened and the later
            # batch_get_into_multi_buffer_ranges returns rc=-600 for every key.
            for req_meta in metadata.requests:
                load_spec = req_meta.load_spec
                if (
                    load_spec is not None
                    and load_spec.can_load
                    and not load_spec.token_len
                ):
                    load_spec.token_len = load_spec.kvpool_cached_tokens

            self._start_layerwise_sessions(metadata.requests)
            self._build_layer_tasks_from_requests(metadata.requests)

    def wait_for_save(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: stores are issued in get_finished() for overlap."""
        pass

    # ============================================================
    # Layerwise KV Cache Methods (Phase 1)
    # ============================================================

    def _build_layer_tasks_from_requests(
        self, requests: list[ReqMeta]
    ) -> None:
        """Build per-layer transfer tasks for each request.

        Called from ``start_load_kv()`` before the model forward pass so that
        ``wait_for_layer_load()`` and ``save_kv_layer()`` have tasks to consume.
        """
        if not self._layerwise_enabled:
            return

        # Clear tasks built for the previous step.
        for layer_id in range(self._num_layers):
            self._layer_save_tasks[layer_id] = []
            self._layer_load_tasks[layer_id] = []

        for req_meta in requests:
            # Save path.
            if req_meta.can_save:
                # Skip tokens already saved (mirrors the legacy save path).
                if hasattr(self.kv_send_thread, "_saved_offset"):
                    save_start = self.kv_send_thread._saved_offset.get(
                        req_meta.req_id, 0
                    )
                else:
                    save_start = 0
                lcm_block_size = self.coord.lcm_block_size
                aligned_token_len = (
                    req_meta.token_len_chunk // lcm_block_size * lcm_block_size
                )
                store_masks = self.coord.store_mask(
                    aligned_token_len,
                    save_start,
                    num_prompt_tokens=req_meta.num_prompt_tokens,
                )

                for group_id, block_ids in enumerate(req_meta.block_ids):
                    db = self.token_dbs[group_id]
                    # Offset each group's chunk start so TP ranks spread evenly.
                    put_step = self._group_tp_replication_factors[group_id]
                    put_step_rank = (self.tp_rank + group_id) % put_step
                    chunks_raw = list(db.process_tokens(
                        req_meta.token_len_chunk,
                        req_meta.block_hashes,
                        mask_num=save_start,
                        chunk_mask=store_masks[group_id],
                        put_step=put_step,
                        put_step_rank=put_step_rank,
                    ))
                    chunks: list[tuple[int, int]] = [(s, e) for s, e, _ in chunks_raw]
                    chunk_hashes: list[BlockHash] = [bh for _, _, bh in chunks_raw]

                    if not chunks:
                        continue

                    # Create one task per physical layer.
                    for physical_layer_id in range(self._num_layers):
                        if self._use_session_api:
                            # Session API: block-level keys + per-layer offsets
                            keys = [db.key_for_block(bh) for bh in chunk_hashes]
                            addrs, sizes, offsets, block_ids_out = (
                                db.prepare_values_for_layer_offset(
                                    chunks, block_ids, physical_layer_id
                                )
                            )
                        else:
                            # Legacy: per-layer @layer:N keys
                            keys = [db.key_for_layer(bh, physical_layer_id) for bh in chunk_hashes]
                            addrs, sizes, block_ids_out = db.prepare_values_for_layer(
                                chunks, block_ids, physical_layer_id
                            )
                            offsets = []

                        task = LayerTransferTask(
                            req_id=req_meta.req_id,
                            group_id=group_id,
                            layer_idx_in_group=physical_layer_id,
                            physical_layer_id=physical_layer_id,
                            key_list=keys,
                            addr_list=addrs,
                            size_list=sizes,
                            dst_offset_list=offsets,
                            block_ids=list(block_ids_out),
                            is_save=True,
                            use_key_major_ranges=self._use_session_api,
                        )
                        self._layer_save_tasks[physical_layer_id].append(task)

            # Load path.
            if req_meta.load_spec and req_meta.load_spec.can_load:
                # Backfill token_len, mirroring the non-layerwise path in
                # get_finished(). LoadSpec is created without token_len; in the
                # non-layerwise path get_finished() fills it from
                # kvpool_cached_tokens, but layerwise builds tasks here, before
                # get_finished() runs.
                if not req_meta.load_spec.token_len:
                    req_meta.load_spec.token_len = req_meta.load_spec.kvpool_cached_tokens
                mask_num = (
                    req_meta.load_spec.vllm_cached_tokens
                    // self.block_size
                    * self.block_size
                )
                load_mask_per_group = self.coord.load_mask(
                    req_meta.block_hashes, req_meta.load_spec.token_len
                )

                for group_id, block_ids in enumerate(req_meta.block_ids):
                    db = self.token_dbs[group_id]
                    mask = load_mask_per_group[group_id]
                    chunks: list[tuple[int, int]] = []
                    chunk_hashes: list[BlockHash] = []
                    # actual_block_ids: per-chunk block ID for address calculation
                    actual_block_ids: list[int] = []
                    for start, end, block_hash in db.process_tokens(
                        req_meta.load_spec.token_len,
                        req_meta.block_hashes,
                        mask_num,
                    ):
                        chunk_idx = start // db.block_size
                        if chunk_idx >= len(mask) or not mask[chunk_idx]:
                            continue
                        chunks.append((start, end))
                        chunk_hashes.append(block_hash)
                        actual_block_ids.append(
                            block_ids[chunk_idx] if chunk_idx < len(block_ids) else -1
                        )

                    if not chunks:
                        continue

                    for physical_layer_id in range(self._num_layers):
                        if self._use_session_api:
                            # Session API: block-level keys + per-layer offsets
                            keys = [db.key_for_block(bh) for bh in chunk_hashes]
                            addrs, sizes, offsets, block_ids_out = (
                                db.prepare_values_for_layer_offset(
                                    chunks, actual_block_ids, physical_layer_id
                                )
                            )
                        else:
                            # Legacy: per-layer @layer:N keys
                            keys = [db.key_for_layer(bh, physical_layer_id) for bh in chunk_hashes]
                            addrs, sizes, block_ids_out = db.prepare_values_for_layer(
                                chunks, actual_block_ids, physical_layer_id
                            )
                            offsets = []

                        # block_ids: the actual block ID for each chunk.
                        block_ids: list[int] = list(block_ids_out)

                        task = LayerTransferTask(
                            req_id=req_meta.req_id,
                            group_id=group_id,
                            layer_idx_in_group=physical_layer_id,
                            physical_layer_id=physical_layer_id,
                            key_list=keys,
                            addr_list=addrs,
                            size_list=sizes,
                            dst_offset_list=offsets,
                            block_ids=block_ids,
                            is_save=False,
                            use_key_major_ranges=self._use_session_api,
                        )
                        self._layer_load_tasks[physical_layer_id].append(task)

    def _submit_ready_layer_loads(self) -> None:
        """Submit the next ready layers for loading (prefetch control).

        While attention computes layer L, submit load tasks for the following
        ``prefetch_layers`` layers. Tasks of the same layer are distributed
        round-robin across recv threads so they never write the same GPU buffer
        concurrently.
        """
        if not self._layerwise_enabled or not self.kv_recv_threads:
            return

        # Submit up to prefetch_layers layers per call.
        submit_count = self._num_prefetch_layers if self._next_load_layer_to_submit == 0 else 1
        submitted = 0

        while (submitted < submit_count
               and self._next_load_layer_to_submit < self._num_layers):
            layer_id = self._next_load_layer_to_submit
            tasks = self._layer_load_tasks.get(layer_id, [])

            if tasks:
                for i, task in enumerate(tasks):
                    # Round-robin: hand each task to a single recv thread.
                    recv_thread = self.kv_recv_threads[
                        i % len(self.kv_recv_threads)
                    ]
                    recv_thread.add_request(task)
                submitted += 1

            self._next_load_layer_to_submit += 1

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
    ) -> None:
        """Save one layer's KV cache to the store.

        Called by ``@maybe_transfer_kv_layer`` after that layer's attention
        forward completes.
        """
        if not self._layerwise_enabled:
            return

        # Extract the layer index from layer_name.
        from vllm.model_executor.models.utils import extract_layer_index
        try:
            layer_id = extract_layer_index(layer_name)
        except Exception:
            # Fallback: parse from layer_name string
            if "model.layers" in layer_name:
                layer_id = int(layer_name.split(".layers.")[1].split(".")[0])
            else:
                logger.warning("Cannot extract layer_id from %s", layer_name)
                return

        # Submit this layer's save tasks to the send thread.
        tasks = self._layer_save_tasks.get(layer_id, [])
        if tasks and self.kv_send_thread:
            for task in tasks:
                self.kv_send_thread.add_request(task)
        elif not tasks:
            # No save tasks (non-producer rank or cold start): mark complete.
            event = self._layer_save_finished_events.get(layer_id)
            if event is not None:
                event.set()

        # Last layer: wait for all saves to complete.
        # Guard: finalize only once per step. If save_kv_layer fires twice for
        # the last layer (e.g. both the @maybe_transfer_kv_layer decorator and
        # an explicit hook), the second call would otherwise hit the
        # _reset_layer_state()-replaced events and hang in _wait_for_all_layer_saves.
        if layer_id == self._num_layers - 1 and not self._save_finalized:
            self._save_finalized = True
            self._wait_for_all_layer_saves()
            # Session API: release get-session leases now that all layers are done.
            self._close_load_sessions_once()
            # Reset state for the next round of requests.
            self._reset_layer_state()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for one layer's KV cache to finish loading from the store.

        Called by ``@maybe_transfer_kv_layer`` before that layer's attention
        forward runs.
        """
        if not self._layerwise_enabled:
            return

        # Extract the layer index from layer_name.
        from vllm.model_executor.models.utils import extract_layer_index
        try:
            layer_id = extract_layer_index(layer_name)
        except Exception:
            if "model.layers" in layer_name:
                layer_id = int(layer_name.split(".layers.")[1].split(".")[0])
            else:
                logger.warning("Cannot extract layer_id from %s", layer_name)
                return

        # Submit prefetch tasks for the following layers.
        self._submit_ready_layer_loads()

        # No load tasks (cold start or full hit): mark complete directly.
        tasks = self._layer_load_tasks.get(layer_id, [])
        if not tasks:
            event = self._layer_load_finished_events.get(layer_id)
            if event is not None:
                event.set()
        else:
            # Wait for this layer's load to complete.
            event = self._layer_load_finished_events.get(layer_id)
            if event is not None and not event.is_set():
                if not event.wait(timeout=10.0):
                    logger.warning("Timeout waiting for layer %d to load", layer_id)
                # Not cleared here; _reset_layer_state() rebuilds all events.

        self._current_load_layer += 1

    def _wait_for_all_layer_saves(self) -> None:
        """Wait for all layers' saves to complete."""
        for layer_id in range(self._num_layers):
            event = self._layer_save_finished_events.get(layer_id)
            if event is not None:
                if not event.wait(timeout=10.0):
                    logger.warning("Timeout waiting for layer %d to save", layer_id)
                # Not cleared here; _reset_layer_state() rebuilds all events.

    def _reset_layer_state(self) -> None:
        """Reset layerwise state for the next round of requests.

        Re-create all event objects (instead of clear()) so a stale set()
        from the previous round's transfer threads cannot be mistaken for the
        current round's completion signal.
        """
        self._current_save_layer = 0
        self._current_load_layer = 0
        self._next_load_layer_to_submit = 0
        self._save_finalized = False

        for layer_id in range(self._num_layers):
            self._layer_save_tasks[layer_id] = []
            self._layer_load_tasks[layer_id] = []

        for layer_id in range(self._num_layers):
            self._layer_save_finished_events[layer_id] = threading.Event()
            self._layer_load_finished_events[layer_id] = threading.Event()

        # Re-sync the events to the transfer threads.
        if self.kv_send_thread is not None and self.kv_send_thread._layerwise_enabled:
            for layer_id in range(self._num_layers):
                self.kv_send_thread.set_layer_finished_event(
                    layer_id, True, self._layer_save_finished_events[layer_id]
                )
        for recv_thread in self.kv_recv_threads:
            if recv_thread._layerwise_enabled:
                for layer_id in range(self._num_layers):
                    recv_thread.set_layer_finished_event(
                        layer_id, False, self._layer_load_finished_events[layer_id]
                    )

        # Session API: reset one-shot guard so the next step can release
        # its load sessions.
        if self._use_session_api:
            self._load_sessions_closed = False
            self._opened_load_keys = []

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """Issue all I/O and get completed send/recv request IDs.

        All load and store I/O requests are issued here (after model
        compute is launched on the compute stream) for better
        compute-I/O overlap.

        Layerwise per-layer tasks are built in start_load_kv() (before
        the model forward pass) so wait_for_layer_load() / save_kv_layer()
        have tasks to consume.  Here we only collect completion.
        """
        if self._capacity_only:
            return set(), set()

        if not self._layerwise_enabled:
            # Issue async loads (bulk path)
            for request in meta.requests:
                load_spec = request.load_spec
                if load_spec is None or not load_spec.can_load:
                    continue
                load_spec.token_len = load_spec.kvpool_cached_tokens
                self.recv_request_queue.put(request)

            assert self.load_async, "load_async must be True for better performance."
            # Issue stores with CUDA event synchronization.
            if self.kv_role in ["kv_producer", "kv_both"]:
                current_event = None
                for request in meta.requests:
                    if request.can_save:
                        current_event = torch.cuda.Event()
                        current_event.record()
                        break

                for request in meta.requests:
                    if not request.can_save:
                        continue
                    request.current_event = current_event
                    assert self.kv_send_thread is not None
                    self.kv_send_thread.add_request(request)

        if self.kv_role in ["kv_producer", "kv_both"]:
            self._close_ended_store_requests(finished_req_ids, meta)

        # Blocks read by a store job are released by the scheduler when the job
        # reports back (see build_connector_worker_meta), so no request ever waits
        # on a `finished_sending` signal to get its blocks back.
        done_sending: set[str] = set()
        done_recving: set[str] = set()
        if self.load_async:
            for recv_thread in self.kv_recv_threads:
                done_recving |= recv_thread.get_and_clear_finished_requests()

        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids: set[int] = set()
        for recv_thread in self.kv_recv_threads:
            block_ids |= recv_thread.get_and_clear_block_ids_with_load_errors()
        return block_ids

    def _record_kv_connector_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def get_kv_connector_stats(self) -> MooncakeStoreConnectorStats | None:
        with self._kv_connector_stats_lock:
            if self.kv_connector_stats.is_empty():
                return None
            kv_connector_stats = self.kv_connector_stats
            self.kv_connector_stats = MooncakeStoreConnectorStats()
            return kv_connector_stats

    def _close_ended_store_requests(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Retire the ledger entries of requests that finished or were preempted.

        An entry may only go once its jobs have drained, because they still read
        the resume offset it owns; a request that comes back after preemption
        then saves from the start rather than from where the last attempt got to.
        """
        assert self.kv_send_thread is not None

        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in finished_req_ids | self.finished_store_req:
            if self.kv_send_thread.stored_requests.get(req_id):
                # Queued jobs still need the resume offset; retire on a later step.
                self.finished_store_req.add(req_id)
            else:
                self.finished_store_req.discard(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

    def build_connector_worker_meta(self) -> MooncakeStoreWorkerMetadata | None:
        if self.kv_send_thread is None:
            return None
        completed_saves = self.kv_send_thread.take_completed_saves()
        if not completed_saves:
            return None
        return MooncakeStoreWorkerMetadata(completed_saves=completed_saves)

    def lookup(self, num_tokens: int, block_hashes: Sequence[BlockHash]) -> int:
        """Check how many prefix tokens exist in the store.

        Checks across all rank-specific key namespaces that may be loaded. A
        hit covering all ``num_tokens`` is re-derived below the request end so
        the last token is recomputed for sampling.

        In layerwise mode, every layer is stored independently under a
        @layer:N key suffix.  A (group, hash) is "present" only when
        ALL layers exist across ALL rank namespaces.
        """
        if self._capacity_only:
            return 0

        token_len = self.coord.align_lookup_length(num_tokens)
        if not block_hashes or token_len <= 0:
            return 0

        # Build per-(group, hash) candidate keys expanded across rank namespaces
        # (and layers when layerwise).
        # candidate_meta stores the (group, hash_bytes) for key slice.
        candidate_keys: list[str] = []
        candidate_meta: list[tuple[int, bytes]] = []
        fine_grained = self.coord.enable_partial_hash_hits
        lookup_masks = None if fine_grained else self.coord.lookup_mask(token_len)
        for g_idx, db in enumerate(self.token_dbs):
            spec_block_size = db.block_size
            key_prefixes = self._lookup_key_prefixes[g_idx]
            if fine_grained:
                max_units = min(len(block_hashes), token_len // self.hash_block_size)
                unit_ids: range | list[int] = range(max_units)
                group_hashes: Sequence[BlockHash] = block_hashes
            else:
                lookup_mask = lookup_masks[g_idx]  # type: ignore[index]
                group_hashes = self.coord.block_hashes_for_spec(
                    block_hashes, self._kv_cache_groups[g_idx].kv_cache_spec
                )
                max_chunks = min(len(group_hashes), cdiv(token_len, spec_block_size))
                mask_limit = (
                    max_chunks
                    if lookup_mask is None
                    else min(max_chunks, len(lookup_mask))
                )
                unit_ids = [
                    chunk_id
                    for chunk_id in range(mask_limit)
                    if lookup_mask is None or lookup_mask[chunk_id]
                ]
            for chunk_id in unit_ids:
                h = group_hashes[chunk_id]
                hash_hex = h.hex()
                for key_prefix in key_prefixes:
                    base_key = PoolKey.build_key_string(key_prefix, hash_hex)
                    if self._layerwise_enabled:
                        if self._use_session_api:
                            # Session API: block-level key covers all layers.
                            # A chunk is present when the block key exists
                            # across ALL rank namespaces (no layer expansion).
                            candidate_keys.append(base_key)
                        else:
                            # Query every layer: a chunk is present only when
                            # ALL layers exist across ALL rank namespaces.
                            for layer_id in range(self._num_layers):
                                candidate_keys.append(
                                    f"{base_key}@layer:{layer_id}"
                                )
                    else:
                        candidate_keys.append(base_key)
                candidate_meta.append((g_idx, bytes(h)))

        if not candidate_keys:
            return 0

        lookup_start = time.perf_counter()
        try:
            res = self.store.batch_is_exist(candidate_keys)
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
            )
        except Exception as e:
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error("Remote connection failed in lookup: %s", e)
            return 0

        # A (group, hash) is "present" only when every namespace that will be
        # loaded has it (per-group count: sharded groups need every rank's
        # shard, replicated groups one namespace per unique KV head).
        # Layerwise non-session mode expands each key across layers, so the
        # count is scaled by num_layers.
        exists_set = set()
        pos = 0
        for g_idx, hash_bytes in candidate_meta:
            count = len(self._lookup_key_prefixes[g_idx])
            if self._layerwise_enabled and not self._use_session_api:
                count *= self._num_layers
            if all(res[pos + j] == 1 for j in range(count)):
                exists_set.add((g_idx, hash_bytes))
            pos += count

        cached_block_pool = ExternalCachedBlockPool(
            self.hash_block_size,
            exists_set,
        )
        _masks, hit_length = self.coord.find_longest_cache_hit(
            block_hashes,
            token_len,
            cached_block_pool,
        )
        if hit_length >= num_tokens:
            usable_length = self.coord.align_lookup_length(num_tokens - 1)
            if usable_length <= 0:
                return 0
            _masks, hit_length = self.coord.find_longest_cache_hit(
                block_hashes,
                usable_length,
                cached_block_pool,
            )
        return hit_length

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []

    def close(self) -> None:
        """Release the MooncakeDistributedStore handle on teardown.

        Closing the store frees its TransferEngine, the registered RDMA
        buffers, and the connection to the master server. Idempotent so it is
        safe to call from both the explicit shutdown path and ``__del__``.
        """
        store = getattr(self, "store", None)
        if store is None:
            return
        self.store = None
        try:
            store.close()
        except Exception as e:
            logger.warning("Error closing MooncakeDistributedStore: %s", e)


# ============================================================
# Lookup Key Server
# ============================================================


class LookupKeyServer:
    """ZMQ server on worker rank 0 for the LookupKey admin channel.

    Handles two request types, tagged at frame 0:
    - ``LOOKUP_MSG``: prefix-cache hit query, returns hit count.
    - ``RESET_MSG``: drains the send thread queue, then runs
      ``store.remove_all(force=True)``. Caller must have paused the
      scheduler first.
    """

    def __init__(
        self,
        store_worker: MooncakeStoreWorker,
        vllm_config: VllmConfig,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self._ipc_path = socket_path.removeprefix("ipc://")
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.store_worker = store_worker
        self.running = True

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                msg_type = bytes(all_frames[0])

                if msg_type == LOOKUP_MSG:
                    num_tokens = int.from_bytes(all_frames[1], byteorder="big")
                    hash_len = int.from_bytes(all_frames[2], byteorder="big")
                    blob = all_frames[3].buffer
                    block_hashes = BlobBlockHashes(blob, hash_len)
                    result = self.store_worker.lookup(num_tokens, block_hashes)
                    self.socket.send(result.to_bytes(4, "big"))

                elif msg_type == RESET_MSG:
                    try:
                        # Drain in-flight puts before wiping the master;
                        # otherwise stale puts can repopulate it post-reset.
                        # Safe across HMA: store.remove_all wipes the underlying
                        # flat key space, clearing every (group_id, hash) entry.
                        if self.store_worker.kv_send_thread is not None:
                            self.store_worker.kv_send_thread.request_queue.join()
                        self.store_worker.store.remove_all(force=True)
                        logger.info("Mooncake store reset via remove_all succeeded.")
                        self.socket.send(RESP_OK)
                    except Exception as e:
                        logger.error("Mooncake remove_all failed: %s", e)
                        self.socket.send(RESP_ERR)

                else:
                    logger.warning(
                        "LookupKeyServer received unknown msg_type: %r",
                        msg_type,
                    )
                    self.socket.send(RESP_ERR)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)


# ============================================================
# Lookup Key Client
# ============================================================


class LookupKeyClient:
    """ZMQ client for the LookupKey admin channel.

    Routes both prefix-cache lookups and admin commands (currently:
    ``reset``) to ``LookupKeyServer`` on worker rank 0. The first frame
    of every request is a named tag from ``protocol.py``.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

        # Async lookup support
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MooncakeLookupClient"
        )
        self.futures: dict[str, Future[int]] = {}

    def _lookup(self, num_tokens: int, block_hashes: list[BlockHash]) -> int:
        hash_len = len(block_hashes[0]) if block_hashes else 0
        all_frames = (
            LOOKUP_MSG,
            num_tokens.to_bytes(4, byteorder="big"),
            hash_len.to_bytes(2, byteorder="big"),
            b"".join(block_hashes),
        )
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        return int.from_bytes(resp, "big")

    def lookup(
        self,
        req_id: str,
        num_tokens: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
    ) -> int | None:
        """If non_block is True, will return None until the result is ready,
        so the caller retries on a later step."""
        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(self._lookup, num_tokens, list(block_hashes))
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            logger.error("Async Mooncake lookup failed for %s: %s", req_id, e)
            return 0
        finally:
            del self.futures[req_id]

    def discard(self, req_id: str) -> None:
        """Drop any cached/in-flight lookup for ``req_id`` (e.g. on abort)."""
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _reset(self) -> bool:
        """Trigger ``store.remove_all(force=True)`` on worker rank 0.

        Ordering assumption: caller MUST ensure no in-flight Mooncake
        lookups or transfers when invoking reset. In RL workflows this
        holds naturally at the step boundary after weight updates and
        rollout drain. Returns True on ACK, False on NACK.
        """
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def reset(self) -> bool:
        return self.executor.submit(self._reset).result()

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    dp_rank = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    hostname = socket.gethostname()
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_host_{hostname}_dp_rank{dp_rank}"
    )
