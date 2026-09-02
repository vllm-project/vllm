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
import math
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
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    PoolKey,
    ReqMeta,
    StoreShardId,
    TailKeyBoundary,
    TPShardedStoreLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # noqa: E501
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
    decode_lookup_response,
    encode_lookup_response,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
    resolve_dcp_kv_cache_spec,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_TENANT_ID = "default"

MOONCAKE_NO_AVAILABLE_HANDLE = -200
_T = TypeVar("_T")


def resolve_store_tp_size(extra_config: dict[str, Any]) -> int | None:
    """Resolve the common Store TP requested by connector config."""
    if extra_config.get("enable_store_tp_lcm") is True:
        prefill_tp_sizes = extra_config.get("prefill_tp_sizes")
        if not isinstance(prefill_tp_sizes, list) or not prefill_tp_sizes:
            return None
        if any(
            type(tp_size) is not int or tp_size <= 0 for tp_size in prefill_tp_sizes
        ):
            return None
        return math.lcm(*prefill_tp_sizes)

    store_tp_size = extra_config.get("store_tp_size")
    return store_tp_size if type(store_tp_size) is int and store_tp_size > 0 else None


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
    return (
        f"vllm-mooncake-store:{prefix}{metadata.model_name}"
        f"{metadata.store_namespace}@{chunk_hash}"
    )


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
    """Base class for async KV cache transfer threads."""

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

    def add_request(self, request: ReqMeta) -> None:
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
        # Retained only after a failed store so retry events can recover the
        # token suffix without full snapshots on the normal path.
        self._retry_token_ids: dict[str, tuple[int, list[int]]] = {}

    def add_request(self, request: ReqMeta) -> None:
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
            self._retry_token_ids.pop(req_id, None)

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

    def _get_retry_token_ids(self, req_meta: ReqMeta) -> tuple[int, list[int]] | None:
        """Return retry state only if this store job is still live."""
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(
                req_meta.req_id, ()
            ):
                return None
            return self._retry_token_ids.get(req_meta.req_id)

    def _update_retry_token_ids(
        self,
        req_meta: ReqMeta,
        save_completed: bool,
        token_ids_start: int,
        event_token_ids: list[int] | None,
    ) -> None:
        """Update retry state without letting a stale job touch a reused ID."""
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(
                req_meta.req_id, ()
            ):
                return
            if save_completed:
                self._retry_token_ids.pop(req_meta.req_id, None)
            elif event_token_ids is not None:
                self._retry_token_ids[req_meta.req_id] = (
                    token_ids_start,
                    event_token_ids,
                )

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

    def _boundary_snapshot_puts(
        self, req_meta: ReqMeta, entries: list[tuple[int, int, int]]
    ) -> list[tuple[str, list[int], list[int], KeyMetadata]]:
        """Puts for committed mamba "align" boundary-state snapshots.

        These are block-aligned boundaries, i.e. exactly what the normal save
        would key — but ``store_mask`` masks mamba groups out of it entirely, so
        this is their *only* writer. The exclusion is not an optimization: the
        normal save resolves a chunk's address as
        ``req_meta.block_ids[g][start // block_size]``, and ``block_ids`` is the
        connector's append-only mirror of the core's per-group table. An
        align-mode table is mutated in place (a superseded state block is freed
        and nulled; speculative blocks relocate), and the connector is never
        told, so a stale mirror entry is indistinguishable from a live one — a
        retry of a failed or pressure-skipped chunk would read a block that now
        belongs to another request.

        Each entry's handed-off block *is* the boundary state and is pinned by
        the core, so it is uploaded under its boundary-end hash key and never
        resolved positionally.
        """
        hash_block_size = self.coord.hash_block_size
        puts: list[tuple[str, list[int], list[int], KeyMetadata]] = []
        for group_id, block_id, boundary in entries:
            if boundary == 0 or block_id == NULL_BLOCK_ID:
                continue
            hash_idx = boundary // hash_block_size - 1
            if hash_idx >= len(req_meta.block_hashes):
                continue
            db = self.token_databases[group_id]
            # Distribute across ranks by the same rule as normal chunks.
            put_step = self.group_put_steps[group_id]
            put_step_rank = (self.tp_rank + group_id) % put_step
            if (boundary // db.block_size - 1) % put_step != put_step_rank:
                continue
            addr, size = db.prepare_value_for_block(block_id)
            puts.append(
                (db.key_for(req_meta.block_hashes[hash_idx]), addr, size, db.metadata)
            )
        return puts

    def _sub_block_tail_puts(
        self, req_meta: ReqMeta, entries: list[tuple[int, int, int]]
    ) -> list[tuple[str, list[int], list[int], KeyMetadata]]:
        """Puts for the request's sub-block partial tail (its last prompt hash
        boundary), so a later request can hit the sub-block prefix.

        Covers every group's blocks from the normal save's lcm floor to the
        boundary: the normal save floors to ``lcm_block_size``, so a
        smaller-block group's full blocks in that gap are never persisted
        elsewhere, and the consumer's lookup needs every group at every probed
        boundary. Full blocks are keyed by their block-end hash and the partial
        boundary block by the boundary sub-hash; a mamba "align" group
        contributes only its boundary block, from the core-provided CoW block.
        """
        boundaries = {boundary for _, _, boundary in entries}
        if len(boundaries) != 1:
            raise ValueError(
                "Sub-block partial-tail offloads for one request must share a boundary"
            )
        boundary = boundaries.pop()
        hash_block_size = self.coord.hash_block_size
        if boundary == 0 or boundary // hash_block_size - 1 >= len(
            req_meta.block_hashes
        ):
            return []

        mamba_offloads = {group_id: block_id for group_id, block_id, _ in entries}
        saved = self._saved_offset.get(req_meta.req_id, 0)
        puts: list[tuple[str, list[int], list[int], KeyMetadata]] = []
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
                if g_idx in mamba_offloads:
                    if valid_end != boundary:
                        # Interior align-mode state positions are null or
                        # stale (the block table is not append-only) and never
                        # valid gap content; only the boundary block is
                        # persisted, from the core-provided hand-off.
                        continue
                    block_id = mamba_offloads[g_idx]
                elif g_idx in self.coord.mamba_group_ids:
                    continue
                elif block_idx < len(group_blocks):
                    block_id = group_blocks[block_idx]
                else:
                    continue
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
                puts.append((db.key_for(key_hash), addr, size, db.metadata))
        return puts

    def _maybe_offload_boundary_states(self, req_meta: ReqMeta) -> bool:
        """Persist connector-pinned mamba "align" boundary states handed off
        for this request, deduped against the store.

        This is every mamba key the connector writes — ``store_mask`` excludes
        mamba groups from the positional normal save, aligned boundaries
        included (see :meth:`_boundary_snapshot_puts`).

        The two entry kinds are keyed and sourced differently, so they are
        prepared separately and put in one batch:

        - block-aligned for its group: a committed boundary-state snapshot,
          the handed-off block itself;
        - not block-aligned: the sub-block CoW partial tail, which also has to
          cover the other groups' blocks in the normal save's lcm gap.

        Returns:
            True when no put is needed or every put succeeds, False otherwise.
        """
        offloads = req_meta.boundary_state_offloads
        if not offloads or not req_meta.block_hashes:
            return True

        snapshots: list[tuple[int, int, int]] = []
        sub_block: list[tuple[int, int, int]] = []
        for group_id, block_id, boundary in offloads:
            entry = (group_id, block_id, boundary)
            if boundary % self.token_databases[group_id].block_size == 0:
                snapshots.append(entry)
            else:
                sub_block.append(entry)

        puts = self._boundary_snapshot_puts(req_meta, snapshots)
        if sub_block and self.coord.enable_partial_hash_hits:
            puts.extend(self._sub_block_tail_puts(req_meta, sub_block))

        if not puts:
            return True
        keys = [key for key, _, _, _ in puts]
        addrs = [addr for _, addr, _, _ in puts]
        sizes = [size for _, _, size, _ in puts]
        group_ids: list[str] | None = (
            [
                _make_mooncake_group_id(metadata, key.rsplit("@", 1)[-1])
                for key, _, _, metadata in puts
            ]
            if self.enable_group_semantics and self.supports_group_ids
            else None
        )
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
                "Failed to check boundary-state keys for request %s: %s",
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
                "Failed to put boundary-state keys for request %s: %s",
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
                "Boundary-state put failed for request %s: %d/%d keys failed "
                "(codes=%s)",
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
                "successful boundary-state batch"
            )
        return True

    def _handle_request(self, req_meta: ReqMeta):
        # The single `finally` is the only way out, so the scheduler releases
        # this job's GPU block references however the job ends.
        save_completed = False
        token_len = 0
        req_id = req_meta.req_id
        event_token_ids = req_meta.token_ids
        token_ids_start = req_meta.token_ids_start
        try:
            # Cache hits are always a multiple of ``lcm_block_size`` tokens,
            # which is also ``store_mask``'s precondition.
            lcm_block_size = self.coord.lcm_block_size
            token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
            block_ids_per_group = req_meta.block_ids
            current_event = req_meta.current_event

            if not self.is_live_store_job(req_meta):
                return

            if self.enable_kv_event:
                retry_token_ids = self._get_retry_token_ids(req_meta)
                if retry_token_ids is not None and event_token_ids is not None:
                    retry_start, retry_ids = retry_token_ids
                    if retry_start + len(retry_ids) == token_ids_start:
                        event_token_ids = retry_ids + event_token_ids
                        token_ids_start = retry_start

            if self._should_skip_request(req_id):
                logger.debug(
                    "Skipping Mooncake store for request %s while CPU/disk "
                    "offloading is under pressure",
                    req_id,
                )
                return

            # Offload the handed-off mamba boundary states (independent of the
            # normal positional save, which may be skipped this step).
            if req_meta.boundary_state_offloads is not None and not (
                self._maybe_offload_boundary_states(req_meta)
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
            event_specs: list[tuple[int, int, int, BlockHash]] | None = (
                [] if self.enable_kv_event else None
            )
            group_indices: list[int] = []
            store_shard_ids: list[StoreShardId] = []
            for g_idx, db in enumerate(self.token_databases):
                # Rotate the stride phase per group to balance load across ranks.
                put_step = self.group_put_steps[g_idx]
                put_step_rank = (self.tp_rank + g_idx) % put_step
                group_blocks = block_ids_per_group[g_idx]
                for start, end, block_hash in db.process_tokens(
                    token_len,
                    req_meta.block_hashes,
                    mask_num=save_start,
                    chunk_mask=store_masks[g_idx],
                    put_step=put_step,
                    put_step_rank=put_step_rank,
                ):
                    block_idx = start // db.block_size
                    group_blocks = block_ids_per_group[g_idx]
                    if block_idx >= len(group_blocks) or (
                        group_blocks[block_idx] == NULL_BLOCK_ID
                    ):
                        logger.debug(
                            "Skipping unavailable Mooncake store source block "
                            "(req=%s, group=%d, block=%d)",
                            req_id,
                            g_idx,
                            block_idx,
                        )
                        continue
                    for store_shard_id in db.store_layout.local_shard_ids:
                        starts.append(start)
                        ends.append(end)
                        keys.append(db.store_layout.key_for(store_shard_id, block_hash))
                        group_indices.append(g_idx)
                        store_shard_ids.append(store_shard_id)
                        if event_specs is not None:
                            event_specs.append((start, end, g_idx, block_hash))

            if not keys:
                self._record_saved(req_meta, token_len)
                save_completed = True
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
                save_completed = True
                return

            if len(missing_indices) != len(keys):
                starts = [starts[i] for i in missing_indices]
                ends = [ends[i] for i in missing_indices]
                keys = [keys[i] for i in missing_indices]
                if event_specs is not None:
                    event_specs = [event_specs[i] for i in missing_indices]
                group_indices = [group_indices[i] for i in missing_indices]
                store_shard_ids = [store_shard_ids[i] for i in missing_indices]

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
            chunks_per_group: list[list[tuple[int, int]]] = [
                [] for _ in self.token_databases
            ]
            shards_per_group: list[list[StoreShardId]] = [
                [] for _ in self.token_databases
            ]
            for start, end, g_idx, store_shard_id in zip(
                starts,
                ends,
                group_indices,
                store_shard_ids,
                strict=True,
            ):
                chunks_per_group[g_idx].append((start, end))
                shards_per_group[g_idx].append(store_shard_id)
            for g_idx, chunks in enumerate(chunks_per_group):
                if not chunks:
                    continue
                db = self.token_databases[g_idx]
                group_addrs, group_sizes, _ = db.store_layout.prepare_values(
                    chunks,
                    block_ids_per_group[g_idx],
                    shards_per_group[g_idx],
                )
                addrs.extend(group_addrs)
                sizes.extend(group_sizes)

            if current_event is not None:
                current_event.synchronize()

            if group_ids is not None:
                assert len(group_ids) == len(keys)
                self.replicate_config.group_ids = group_ids

            failed_indices: set[int] = set()
            put_had_exception = False
            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
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
                put_had_exception = True
            else:
                failed_indices = {i for i, value in enumerate(res) if value < 0}
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed_indices else "ok",
                    num_failed_keys=len(failed_indices),
                )
                failed_codes = {res[i] for i in failed_indices}
                if failed_indices:
                    logger.warning(
                        "batch_put failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d), first_key=%s",
                        len(failed_indices),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        keys[0],
                    )
                if (
                    MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                    and not self._mark_request_skipped_for_pressure(req_meta)
                ):
                    logger.warning(
                        "Detected Mooncake CPU/disk offloading pressure "
                        "(NO_AVAILABLE_HANDLE); skipping future store "
                        "batches for request %s until a later store batch succeeds",
                        req_id,
                    )

            if not put_had_exception and not failed_indices:
                self._record_saved(req_meta, token_len)
                save_completed = True
                if self._clear_store_pressure():
                    logger.info(
                        "Mooncake CPU/disk offloading pressure cleared "
                        "after a successful store batch"
                    )

            stored_events: list[BlockStored] = []
            if self.enable_kv_event and not put_had_exception:
                assert event_specs is not None
                # BlockStored is a logical-block event, while one block may map
                # to several Store shards. Emit once only after every missing
                # shard for that block succeeded. Shards that already existed
                # were removed before this mapping and are already satisfied.
                indices_by_event: dict[tuple[int, int, int, BlockHash], list[int]] = {}
                for index, event_spec in enumerate(event_specs):
                    indices_by_event.setdefault(event_spec, []).append(index)

                token_ids_end = token_ids_start + len(event_token_ids or ())
                for (
                    s,
                    end,
                    g_idx,
                    block_hash,
                ), event_indices in indices_by_event.items():
                    if any(index in failed_indices for index in event_indices):
                        continue
                    db = self.token_databases[g_idx]
                    token_ids = (
                        event_token_ids[s - token_ids_start : end - token_ids_start]
                        if event_token_ids is not None
                        and token_ids_start <= s
                        and end <= token_ids_end
                        else []
                    )
                    stored_events.append(
                        BlockStored(
                            block_hashes=[maybe_convert_block_hash(block_hash)],
                            # Store filtering can separate adjacent request
                            # blocks, so derive the predecessor from the request.
                            parent_block_hash=(
                                maybe_convert_block_hash(
                                    req_meta.block_hashes[s // db.hash_block_size - 1]
                                )
                                if s > 0
                                else None
                            ),
                            token_ids=token_ids,
                            block_size=db.block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                            group_idx=g_idx,
                        )
                    )

            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)
        finally:
            if self.enable_kv_event and token_len:
                self._update_retry_token_ids(
                    req_meta,
                    save_completed,
                    token_ids_start,
                    event_token_ids,
                )
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

    def _add_load_error_block_ids(self, block_ids: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def _handle_request(self, req_meta: ReqMeta):
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
        tail_key_boundaries = {
            boundary.group_id: boundary.num_tokens
            for boundary in (
                req_meta.load_spec.tail_key_boundaries  # type: ignore[union-attr]
            )
        }

        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        for g_idx, db in enumerate(self.token_databases):
            mask = load_mask_per_group[g_idx]
            chunks: list[tuple[int, int]] = []
            store_shard_ids: list[StoreShardId] = []
            for start, end, block_hash in db.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                boundary_tokens = (
                    tail_key_boundaries.get(g_idx) if end == token_len else None
                )
                if boundary_tokens is not None:
                    block_hash = req_meta.block_hashes[
                        boundary_tokens // db.hash_block_size - 1
                    ]
                for store_shard_id in db.store_layout.local_shard_ids:
                    key_list.append(db.store_layout.key_for(store_shard_id, block_hash))
                    chunks.append((start, end))
                    store_shard_ids.append(store_shard_id)
            g_addrs, g_sizes, g_block_ids = db.store_layout.prepare_values(
                chunks,
                req_meta.block_ids[g_idx],
                store_shard_ids,
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

        load_batches = [
            (
                key_list_c,
                addr_list_c,
                size_list_c,
                block_id_list_c,
            )
        ]
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
                        (
                            batch_keys,
                            batch_addrs,
                            batch_sizes,
                            batch_block_ids,
                        )
                    )
                    block_id_offset = next_block_id_offset

        current_batch_keys: list[str] = key_list_c
        current_batch_block_ids: list[int] = block_id_list_c
        batch_bytes = 0
        try:
            for (
                batch_keys,
                batch_addrs,
                batch_sizes,
                batch_block_ids,
            ) in load_batches:
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
        kv_role = vllm_config.kv_transfer_config.kv_role
        assert kv_role is not None
        self.kv_role = kv_role
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.can_put = self.kv_role in ("kv_producer", "kv_both") or (
            extra_config.get("save_decode_cache", False)
        )
        self.load_async = extra_config.get("load_async", True)
        # Mirrors MooncakeStoreConnector._capacity_only.
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not extra_config.get("enable_lookup", True)
            and not self.can_put
        )
        self.cache_config = vllm_config.cache_config
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.num_layers = model_config.get_num_layers(parallel_config)

        self.num_kv_head = model_config.get_total_num_kv_heads()

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
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

        self._kv_cache_groups = [
            dataclasses.replace(
                group,
                kv_cache_spec=resolve_dcp_kv_cache_spec(
                    group.kv_cache_spec, self.dcp_size
                ),
            )
            for group in kv_cache_config.transfer_groups
        ]
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle_block_drop = bool(
            spec_cfg.use_eagle_block_drop()
            if spec_cfg is not None
            and callable(getattr(spec_cfg, "use_eagle_block_drop", None))
            else False
        )
        self.coord = MooncakeStoreCoordinator(
            self._kv_cache_groups,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle_block_drop,
            retention_interval=kv_cache_config.prefix_cache_retention_interval,
            dcp_world_size=self.dcp_size,
        )
        self.store_tp_size, store_namespace, store_layout_cls = (
            self._select_store_layout(extra_config)
        )
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
            store_namespace=store_namespace,
        )
        self._group_tp_replication_factors: tuple[int, ...] = (
            self._compute_group_tp_replication_factors()
        )
        self.token_dbs = self._build_token_databases(metadata, store_layout_cls)
        self._init_lookup_key_prefixes()

    def _supports_tp_sharded_store_layout(
        self,
        layout_cls: type[TPShardedStoreLayout] | None,
        extra_config: dict[str, Any],
    ) -> bool:
        if (
            layout_cls is None
            or self.pcp_size != 1
            or self.dcp_size != 1
            or len(self._kv_cache_groups) != 1
        ):
            return False

        # Subclasses may use different sharding or cache-lifetime semantics.
        return (
            type(self._kv_cache_groups[0].kv_cache_spec) is FullAttentionSpec
            and str(extra_config.get("enable_cross_layers_blocks", "False")).lower()
            != "true"
        )

    def _select_store_layout(
        self, extra_config: dict[str, Any]
    ) -> tuple[int | None, str, type[TPShardedStoreLayout] | None]:
        """Select the opt-in TP layout and its Store namespace."""
        lcm_store_tp_enabled = extra_config.get("enable_store_tp_lcm") is True
        store_tp_requested = (
            lcm_store_tp_enabled or extra_config.get("store_tp_size") is not None
        )
        if not store_tp_requested:
            return None, "", None

        requested_store_tp_size = resolve_store_tp_size(extra_config)
        cache_layout = self.cache_config.get_resolved_kv_cache_layout()
        layout_cls: type[TPShardedStoreLayout] | None = {
            KVCacheLayout.LBHNC: LBHNCStoreLayout,
            KVCacheLayout.LBNHC: LBNHCStoreLayout,
        }.get(cache_layout)

        if (
            requested_store_tp_size is not None
            and requested_store_tp_size >= self.tp_size
            and requested_store_tp_size % self.tp_size == 0
            and self._supports_tp_sharded_store_layout(layout_cls, extra_config)
        ):
            assert layout_cls is not None
            if self.num_kv_head % requested_store_tp_size == 0:
                if layout_cls is LBNHCStoreLayout:
                    logger.warning_once(
                        "Mooncake Store TP sharding is using the LBNHC (NHC) "
                        "KV cache layout, which creates many transfer segments "
                        "and may significantly reduce PUT/GET performance. Use "
                        "LBHNC (HNC) when supported by the attention backend."
                    )
                return (
                    requested_store_tp_size,
                    layout_cls.shared_namespace(requested_store_tp_size, self.pp_size),
                    layout_cls,
                )
            if self.num_kv_head == 1:
                logger.info(
                    "Mooncake heterogeneous-TP store sharing uses the replicated "
                    "MQA layout for store_tp_size=%d",
                    requested_store_tp_size,
                )
                return (
                    None,
                    f"@store_pp:{self.pp_size}@store_format:tp_shared_mqa",
                    None,
                )

        store_namespace = (
            f"@store_pp:{self.pp_size}@store_format:"
            f"rank_local_tp{self.tp_size}_layout_{cache_layout.name}"
        )
        requested_topology = (
            extra_config.get("prefill_tp_sizes")
            if lcm_store_tp_enabled
            else extra_config.get("store_tp_size")
        )
        logger.warning(
            "Mooncake heterogeneous-TP store sharing is disabled for "
            "Store TP configuration %r with KV layout %s; using a "
            "compatibility-namespaced rank-local store layout",
            requested_topology,
            cache_layout,
        )
        return None, store_namespace, None

    def _build_token_databases(
        self,
        metadata: KeyMetadata,
        layout_cls: type[TPShardedStoreLayout] | None,
    ) -> list[ChunkedTokenDatabase]:
        """Construct token databases and their Store layouts."""
        token_dbs: list[ChunkedTokenDatabase] = []
        for group_idx, group in enumerate(self._kv_cache_groups):
            group_tp_rank = self.tp_rank
            if layout_cls is None:
                group_tp_rank //= self._group_tp_replication_factors[group_idx]
            group_metadata = dataclasses.replace(
                metadata,
                group_id=group_idx,
                tp_rank=group_tp_rank,
            )
            store_layout: TPShardedStoreLayout | None = None
            if layout_cls is not None:
                assert self.store_tp_size is not None
                store_layout = layout_cls(
                    group_metadata,
                    group.kv_cache_spec.block_size,
                    self.hash_block_size,
                    local_tp_size=self.tp_size,
                    store_tp_size=self.store_tp_size,
                    tp_rank=self.tp_rank,
                    num_kv_heads=self.num_kv_head,
                )
            token_dbs.append(
                ChunkedTokenDatabase(
                    group_metadata,
                    group.kv_cache_spec.block_size,
                    hash_block_size=self.hash_block_size,
                    store_layout=store_layout,
                )
            )
        return token_dbs

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
            db.store_layout.lookup_key_prefixes(
                rank_namespaces(self._group_tp_replication_factors[g_idx])
            )
            for g_idx, db in enumerate(self.token_dbs)
        )

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register KV cache tensors and start transfer threads."""
        if self._capacity_only:
            return
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        seen_storage_ptrs: set[int] = set()
        cache_tensors: list[torch.Tensor] = []

        for cache in kv_caches.values():
            cache = group_kernel_blocks(cache, self.num_blocks)
            cache_tensors.append(cache)
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            region_len = cache_storage.nbytes()

            if base_addr not in seen_storage_ptrs:
                seen_storage_ptrs.add(base_addr)
                ret = self.store.register_buffer(base_addr, region_len)
                if ret != 0:
                    logger.error(
                        "register_buffer failed for addr %#x len %d: %d",
                        base_addr,
                        region_len,
                        ret,
                    )

        logger.info(
            "Registered KV caches: num_groups=%d, num_tensors=%d, num_blocks=%d",
            len(self.token_dbs),
            len(cache_tensors),
            self.num_blocks,
        )

        for db in self.token_dbs:
            db.store_layout.register_kv_caches(cache_tensors, self.num_blocks)

        # Start transfer threads
        if self.can_put:
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
            recv_thread.name = f"KVCacheStoreRecvingThread-{i}"
            recv_thread.start()
            self.kv_recv_threads.append(recv_thread)
            ready_events_recving.append(ready_event_recving)
        for ready_event_recving in ready_events_recving:
            ready_event_recving.wait()
        logger.info(
            "Started %d Mooncake KV-load receive thread(s)", self.num_recv_threads
        )

    def start_load_kv(self, metadata: MooncakeStoreConnectorMetadata):
        """Issue async loads.

        Runs after the forward launch on steps without sync loads
        (SchedulerOutput.has_sync_kv_loads), keeping load submission off
        the critical path while preserving compute-I/O overlap.
        """
        if self._capacity_only:
            return

        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue

            load_spec.token_len = load_spec.kvpool_cached_tokens
            self.recv_request_queue.put(request)

        assert self.load_async, "load_async must be True for better performance."

    def wait_for_save(self, metadata: MooncakeStoreConnectorMetadata):
        """Issue async stores with CUDA event synchronization.

        Runs after the forward launch for compute-I/O overlap.
        """
        if self._capacity_only or not self.can_put:
            return

        current_event = None
        for request in metadata.requests:
            if request.can_save:
                current_event = torch.cuda.Event()
                current_event.record()
                break

        for request in metadata.requests:
            if not request.can_save:
                continue
            request.current_event = current_event
            assert self.kv_send_thread is not None
            self.kv_send_thread.add_request(request)

    def get_finished(
        self, finished_req_ids: set[str], meta: MooncakeStoreConnectorMetadata
    ) -> tuple[set[str], set[str]]:
        """Get completed send/recv request IDs.

        Loads are issued in start_load_kv() and stores in wait_for_save().
        """
        if self._capacity_only:
            return set(), set()

        if self.can_put:
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

    def lookup(
        self, num_tokens: int, block_hashes: Sequence[BlockHash]
    ) -> MooncakeLookupResult:
        """Check how many prefix tokens exist in the store.

        Checks across all rank-specific key namespaces that may be loaded. A
        hit covering all ``num_tokens`` is re-derived below the request end so
        the last token is recomputed for sampling.
        """
        if self._capacity_only:
            return MooncakeLookupResult(0)

        token_len = self.coord.align_lookup_length(num_tokens)
        if not block_hashes or token_len <= 0:
            return MooncakeLookupResult(0)

        # Build per-(group, hash) candidate keys expanded across rank namespaces.
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
                    candidate_keys.append(
                        PoolKey.build_key_string(key_prefix, hash_hex)
                    )
                candidate_meta.append((g_idx, bytes(h)))

        if not candidate_keys:
            return MooncakeLookupResult(0)

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
            return MooncakeLookupResult(0)

        # A (group, hash) is "present" only when every namespace that will be
        # loaded has it (per-group count: sharded groups need every rank's
        # shard, replicated groups one namespace per unique KV head).
        exists_set = set()
        pos = 0
        for g_idx, hash_bytes in candidate_meta:
            count = len(self._lookup_key_prefixes[g_idx])
            if all(res[pos + j] == 1 for j in range(count)):
                exists_set.add((g_idx, hash_bytes))
            pos += count

        cached_block_pool = ExternalCachedBlockPool(
            self.hash_block_size,
            exists_set,
        )
        _, hit_length = self.coord.find_longest_cache_hit(
            block_hashes,
            token_len,
            cached_block_pool,
        )
        if hit_length >= num_tokens:
            usable_length = self.coord.align_lookup_length(num_tokens - 1)
            if usable_length <= 0:
                return MooncakeLookupResult(0)
            _, hit_length = self.coord.find_longest_cache_hit(
                block_hashes,
                usable_length,
                cached_block_pool,
            )
        return MooncakeLookupResult(
            hit_length,
            self._tail_key_boundaries(
                block_hashes,
                hit_length,
                cached_block_pool,
            ),
        )

    def _tail_key_boundaries(
        self,
        block_hashes: Sequence[BlockHash],
        hit_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[TailKeyBoundary, ...]:
        """Return the hash boundary used to store each group's tail block.

        With fine-grained prefix matching, ``hit_length`` may fall within a
        physical cache block and may not align with the hash boundary used to
        store that block. For each KV-cache group, return the token boundary
        whose hash was used as the store key.
        """
        if hit_length <= 0:
            return ()

        boundaries = []
        hit_boundary_hash_idx = hit_length // self.hash_block_size - 1
        for group_id, db in enumerate(self.token_dbs):
            chunk_id = cdiv(hit_length, db.block_size) - 1
            boundary_tokens = hit_length
            contains_hit_boundary = cached_block_pool.contains(
                group_id, block_hashes[hit_boundary_hash_idx]
            )
            if not self.coord.enable_partial_hash_hits:
                assert contains_hit_boundary
            if not contains_hit_boundary:
                next_chunk_hash_idx = min(
                    (chunk_id + 1) * db.block_size // self.hash_block_size,
                    len(block_hashes),
                )
                for hash_idx in range(hit_boundary_hash_idx + 1, next_chunk_hash_idx):
                    if cached_block_pool.contains(group_id, block_hashes[hash_idx]):
                        boundary_tokens = (hash_idx + 1) * self.hash_block_size
                        break
                else:
                    raise AssertionError(
                        f"No tail key found for cache group {group_id} at "
                        f"hit length {hit_length}"
                    )
            boundaries.append(TailKeyBoundary(group_id, boundary_tokens))
        return tuple(boundaries)

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []

    @property
    def group_tp_replication_factors(self) -> tuple[int, ...]:
        return self._group_tp_replication_factors

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
    - ``LOOKUP_MSG``: prefix-cache hit query, returns its load plan.
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
                    self.socket.send(encode_lookup_response(result))

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
        self.futures: dict[str, Future[MooncakeLookupResult]] = {}

    def _lookup(
        self, num_tokens: int, block_hashes: list[BlockHash]
    ) -> MooncakeLookupResult:
        hash_len = len(block_hashes[0]) if block_hashes else 0
        all_frames = (
            LOOKUP_MSG,
            num_tokens.to_bytes(4, byteorder="big"),
            hash_len.to_bytes(2, byteorder="big"),
            b"".join(block_hashes),
        )
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        return decode_lookup_response(resp)

    def lookup(
        self,
        req_id: str,
        num_tokens: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
    ) -> MooncakeLookupResult | None:
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
            return MooncakeLookupResult(0)
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
