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
from collections import defaultdict
from collections.abc import Callable
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
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
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
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from .metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB

MOONCAKE_NO_AVAILABLE_HANDLE = -200
_T = TypeVar("_T")


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]


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
        )

    @staticmethod
    def load_from_config() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


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
    ):
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self._record_operation_cb = record_operation
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
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
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)

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
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
        replicate_config: Any = None,
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
        self.put_step = put_step
        self.coord = coord
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.enable_kv_event = enable_kv_event
        # Caller always passes a non-None ReplicateConfig — see
        # MooncakeStoreWorker.__init__ where store_replicate_config is built.
        self.replicate_config = replicate_config

        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._skip_store_requests.discard(req_id)

    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _mark_request_skipped_for_pressure(self, req_id: str) -> bool:
        with self.done_task_lock:
            already_skipped = req_id in self._skip_store_requests
            self._store_pressure_active = True
            self._skip_store_requests.add(req_id)
        return already_skipped

    def _clear_store_pressure(self) -> bool:
        with self.done_task_lock:
            if not self._store_pressure_active and not self._skip_store_requests:
                return False
            self._store_pressure_active = False
            self._skip_store_requests.clear()
        return True

    def _handle_request(self, req_meta: ReqMeta):
        # Cache hits are always a multiple of ``lcm_block_size`` tokens, which
        # is also ``store_mask``'s precondition.
        lcm_block_size = self.coord.lcm_block_size
        token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
        block_ids_per_group = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        # Decrement the in-flight counter and signal task_done() in `finally`
        # so the scheduler can release the GPU blocks it pinned for this
        # request (via `delay_free_blocks`) even when the store path raises.
        try:
            if token_len == 0:
                return
            if self._should_skip_request(req_id):
                logger.debug(
                    "Skipping Mooncake store for request %s while CPU/disk "
                    "offloading is under pressure",
                    req_id,
                )
                return

            # Within each lcm region only per-spec relevant chunks are loaded
            # (e.g., SWA or linear attn), so mask out irrelevant chunks
            store_masks = self.coord.store_mask(
                token_len, num_prompt_tokens=req_meta.num_prompt_tokens
            )
            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            block_hashes: list[BlockHash] = []
            group_indices: list[int] = []
            for g_idx, db in enumerate(self.token_databases):
                mask = store_masks[g_idx]
                for chunk_idx, (start, end, key) in enumerate(
                    db.process_tokens(token_len, req_meta.block_hashes)
                ):
                    if chunk_idx >= len(mask) or not mask[chunk_idx]:
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_string())
                    block_hashes.append(BlockHash(bytes.fromhex(key.chunk_hash)))
                    group_indices.append(g_idx)

            # Apply put_step striding for TP
            sl = slice(self.tp_rank % self.put_step, None, self.put_step)
            starts = starts[sl]
            ends = ends[sl]
            keys = keys[sl]
            block_hashes = block_hashes[sl]
            group_indices = group_indices[sl]

            if not keys:
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
                return

            starts = [starts[i] for i in missing_indices]
            ends = [ends[i] for i in missing_indices]
            keys = [keys[i] for i in missing_indices]
            block_hashes = [block_hashes[i] for i in missing_indices]
            group_indices = [group_indices[i] for i in missing_indices]

            logger.debug(
                "Storing KV cache for %d blocks (groups=%s) for request %s",
                len(keys),
                set(group_indices),
                req_id,
            )

            addrs: list[list[int]] = []
            sizes: list[list[int]] = []
            stored_events: list[BlockStored] = []
            # parent_block_hash chains live within a group, not across.
            prev_key_per_group: dict[int, Any] = {}
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]
                addr, size, _ = db.prepare_value(s, e, block_ids_per_group[g_idx])
                addrs.append(addr)
                sizes.append(size)

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
                        and not self._mark_request_skipped_for_pressure(req_id)
                    ):
                        logger.warning(
                            "Detected Mooncake CPU/disk offloading pressure "
                            "(NO_AVAILABLE_HANDLE); skipping future store "
                            "batches for request %s until a later store "
                            "batch succeeds",
                            req_id,
                        )
                elif self._clear_store_pressure():
                    logger.info(
                        "Mooncake CPU/disk offloading pressure cleared after a "
                        "successful store batch"
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
            self.dec_stored_request(req_id)
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
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
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

        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        for g_idx, db in enumerate(self.token_databases):
            mask = load_mask_per_group[g_idx]
            for start, end, key in db.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                addr, size, block_id = db.prepare_value(
                    start, end, req_meta.block_ids[g_idx]
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)

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

        self.dp_rank = get_mooncake_dp_engine_index(parallel_config)
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
        self.cache_config = vllm_config.cache_config
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.num_layers = model_config.get_num_layers(parallel_config)

        self.use_mla = False
        if (
            hasattr(model_config, "use_mla")
            and isinstance(model_config.use_mla, bool)
            and model_config.use_mla
        ):
            self.use_mla = True

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        self.metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            tp_rank=self.head_or_tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
            cache_prefix=str(
                vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                    "cache_prefix", ""
                )
            ),
        )

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
        ret = self.store.setup(
            local_hostname,
            store_config.metadata_server,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
        )
        if ret != 0:
            msg = "Initialize MooncakeDistributedStore failed."
            logger.error(msg)
            raise RuntimeError(msg)

        preferred_segment = rdma_utils.get_configured_preferred_segment(extra_config)
        self.preferred_segment = preferred_segment
        self.store_replicate_config = ReplicateConfig()
        if preferred_segment is not None:
            self.store_replicate_config.preferred_segment = preferred_segment

        logger.info(
            "Mooncake mode=%s (global_segment_size=%d, local_buffer_size=%d, "
            "preferred_segment=%s, enable_offload=%s)",
            store_config.mode,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            preferred_segment or "<none>",
            store_config.enable_offload,
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
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.finished_store_req: set[str] = set()
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()

        self._kv_cache_config = kv_cache_config
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
            retention_interval=envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL,
        )
        # One ChunkedTokenDatabase per group; addresses populated in
        # register_kv_caches once the kv-cache layout is known.
        self.token_dbs: list[ChunkedTokenDatabase] = [
            ChunkedTokenDatabase(
                dataclasses.replace(self.metadata, group_id=g_idx),
                g.kv_cache_spec.block_size,
                hash_block_size=self.hash_block_size,
            )
            for g_idx, g in enumerate(self._kv_cache_groups)
        ]

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

        # Start transfer threads
        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
                self.store_replicate_config,
                record_operation=self._record_kv_connector_operation,
            )
            self.kv_send_thread.start()

        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.store,
            self.coord,
            self.token_dbs,
            self.block_size,
            self.tp_rank,
            ready_event_recving,
            disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
            record_operation=self._record_kv_connector_operation,
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()

    def start_load_kv(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: loads are issued in get_finished() for overlap."""
        pass

    def wait_for_save(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: stores are issued in get_finished() for overlap."""
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """Issue all I/O and get completed send/recv request IDs.

        All load and store I/O requests are issued here (after model
        compute is launched on the compute stream) for better
        compute-I/O overlap.
        """
        # Issue async loads
        for request in meta.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue

            load_spec.token_len = load_spec.kvpool_cached_tokens

            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)

        assert self.load_async, "load_async must be True for better performance."
        # Issue stores with CUDA event synchronization
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
                self.kv_send_thread.add_stored_request(request.req_id)
                self.kv_send_thread.add_request(request)

        # Check completion of previously queued transfers
        done_sending = (
            self._get_and_clear_finished_sending(finished_req_ids, meta)
            if self.kv_role in ["kv_producer", "kv_both"]
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()
            if self.load_async and self.kv_recv_thread is not None
            else set()
        )

        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_recv_thread is None:
            return set()
        return self.kv_recv_thread.get_and_clear_block_ids_with_load_errors()

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

    def _get_and_clear_finished_sending(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> set[str]:
        assert self.kv_send_thread is not None
        finished_sending: set[str] = set()

        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in self.kv_send_thread.stored_requests.copy():
            if (
                self.kv_send_thread.stored_requests[req_id] == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        """Check how many prefix tokens exist in the store.

        Checks across all TP ranks and PP ranks.
        """
        if not block_hashes or token_len <= 0:
            return 0

        # Build per-(group, hash) candidate keys expanded across TP/PP.
        # candidate_meta[i] is the (group_id, hash_bytes) for candidate_keys[i].
        candidate_keys: list[str] = []
        candidate_meta: list[tuple[int, bytes]] = []
        tp_count = min(self.tp_size, self.num_kv_head)
        for g_idx, db in enumerate(self.token_dbs):
            spec_block_size = db.block_size
            group_hashes = self.coord.block_hashes_for_spec(
                block_hashes, self._kv_cache_groups[g_idx].kv_cache_spec
            )
            for chunk_id, h in enumerate(group_hashes):
                start_idx = chunk_id * spec_block_size
                if start_idx >= token_len:
                    break
                for tp in range(tp_count):
                    for pp in range(self.pp_size):
                        md = dataclasses.replace(db.metadata, tp_rank=tp, pp_rank=pp)
                        candidate_keys.append(PoolKey(md, h.hex()).to_string())
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

        # A (group, hash) is "present" only when every TP*PP rank has it.
        expected_per_key = max(1, tp_count * self.pp_size)
        present_count: dict[tuple[int, bytes], int] = {}
        for gh, exists in zip(candidate_meta, res, strict=True):
            if exists == 1:
                present_count[gh] = present_count.get(gh, 0) + 1
        exists_set = {gh for gh, c in present_count.items() if c >= expected_per_key}

        _masks, hit_length = self.coord.find_longest_cache_hit(
            block_hashes, token_len, ExternalCachedBlockPool(exists_set)
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
        self.decoder = MsgpackDecoder()
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
                    token_len = int.from_bytes(all_frames[1], byteorder="big")
                    hash_frames = all_frames[2:]
                    hashes_str = self.decoder.decode(hash_frames)
                    block_hashes = [BlockHash(bytes.fromhex(s)) for s in hashes_str]
                    result = self.store_worker.lookup(token_len, block_hashes)
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
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        all_frames = [LOOKUP_MSG, token_len_bytes] + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def reset(self) -> bool:
        """Trigger ``store.remove_all(force=True)`` on worker rank 0.

        Ordering assumption: caller MUST ensure no in-flight Mooncake
        lookups or transfers when invoking reset. In RL workflows this
        holds naturally at the step boundary after weight updates and
        rollout drain. Returns True on ACK, False on NACK.
        """
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    dp_rank = get_mooncake_dp_engine_index(vllm_config.parallel_config)
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
