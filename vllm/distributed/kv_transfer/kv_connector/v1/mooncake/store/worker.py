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

import json
import os
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

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
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)


def _unwrap_block_ids(
    block_ids: list[int] | tuple[list[int], ...],
    group_idx: int = 0,
) -> list[int]:
    """Extract a single group's block IDs from HMA or non-HMA format."""
    if isinstance(block_ids, tuple):
        return block_ids[group_idx]
    return block_ids


DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
MOONCAKE_NO_AVAILABLE_HANDLE = -200


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore."""

    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address", ""),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
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


# ============================================================
# Transfer Threads
# ============================================================


class KVTransferThread(threading.Thread):
    """Base class for async KV cache transfer threads."""

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_database = token_database
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
        group_token_databases: list[ChunkedTokenDatabase]
        | ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        group_put_steps: list[int],
        group_sw_blocks: list[int],
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
    ):
        if isinstance(group_token_databases, ChunkedTokenDatabase):
            group_token_databases = [group_token_databases]
        super().__init__(
            store,
            group_token_databases[0]
            if group_token_databases
            else ChunkedTokenDatabase(
                KeyMetadata("", 0, 0, 0, 0), block_size
            ),
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
        )
        self.group_token_databases = group_token_databases
        self.group_put_steps = group_put_steps
        self.group_sw_blocks = group_sw_blocks
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.enable_kv_event = enable_kv_event

        # Pause store requests when CPU offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()

    def add_stored_request(self, req_id: str, count: int = 1):
        with self.done_task_lock:
            self.stored_requests[req_id] += count

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def dec_stored_request_by(self, req_id: str, count: int):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= count

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

    def _active_group_indices(
        self, block_ids: list[int] | tuple[list[int], ...]
    ) -> list[int]:
        if isinstance(block_ids, list):
            return [0]
        return list(range(len(block_ids)))

    def _handle_request(self, req_meta: ReqMeta):
        req_id = req_meta.req_id
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        if self._should_skip_request(req_id):
            logger.debug(
                "Skipping Mooncake store for request %s while CPU offloading "
                "is under pressure",
                req_id,
            )
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        for group_idx in self._active_group_indices(req_meta.block_ids):
            token_database = self.group_token_databases[group_idx]
            if not token_database.kv_caches_base_addr:
                continue
            self._handle_request_for_group(req_meta, group_idx, token_database)

        self.dec_stored_request(req_id)
        self.request_queue.task_done()

    def _handle_request_for_group(
        self,
        req_meta: ReqMeta,
        group_idx: int,
        token_database: ChunkedTokenDatabase,
    ):
        token_len = req_meta.token_len_chunk
        block_ids = _unwrap_block_ids(req_meta.block_ids, group_idx)
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        # Compute SWA mask: skip blocks outside the sliding window.
        sw_blocks = self.group_sw_blocks[group_idx]
        swa_mask = 0
        if sw_blocks > 0:
            total_blocks = (
                token_len + token_database.block_size - 1
            ) // token_database.block_size
            if total_blocks > sw_blocks:
                swa_mask = (total_blocks - sw_blocks) * token_database.block_size

        starts = []
        ends = []
        keys = []
        block_hashes: list[BlockHash] = []
        for index, (start, end, key) in enumerate(
            token_database.process_tokens(
                token_len, req_meta.block_hashes, swa_mask
            )
        ):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])

        # Apply put_step striding for TP
        put_step = self.group_put_steps[group_idx]
        starts = starts[self.tp_rank % put_step :: put_step]
        ends = ends[self.tp_rank % put_step :: put_step]
        keys = keys[self.tp_rank % put_step :: put_step]
        block_hashes = block_hashes[self.tp_rank % put_step :: put_step]

        if not keys:
            return

        # Check which blocks already exist (dedup)
        exists_states = self.store.batch_is_exist(keys)
        missing_indices = [i for i, exists in enumerate(exists_states) if exists != 1]

        if not missing_indices:
            return

        starts = [starts[i] for i in missing_indices]
        ends = [ends[i] for i in missing_indices]
        keys = [keys[i] for i in missing_indices]
        block_hashes = [block_hashes[i] for i in missing_indices]

        logger.debug(
            "Storing KV cache for %d out of %d blocks "
            "(missing_count=%d) for request %s group %d",
            len(keys),
            token_len // token_database.block_size,
            len(missing_indices),
            req_id,
            group_idx,
        )

        addrs = []
        sizes = []
        stored_events: list[BlockStored] = []
        prev_key = None
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

        for index, start in enumerate(starts):
            addr, size, _ = token_database.prepare_value(
                start, ends[index], block_ids
            )
            addrs.append(addr)
            sizes.append(size)

            if self.enable_kv_event:
                token_ids = (
                    req_meta.token_ids[start : ends[index]]
                    if req_meta.token_ids is not None
                    else None
                )
                stored_event = BlockStored(
                    block_hashes=[new_block_hashes[index]],
                    parent_block_hash=prev_key,
                    token_ids=token_ids,
                    block_size=req_meta.original_block_size,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                stored_events.append(stored_event)
                prev_key = new_block_hashes[index]

        if current_event is not None:
            current_event.synchronize()

        try:
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes)
            failed = [i for i, v in enumerate(res) if v < 0]
            if failed:
                # Compute total bytes attempted for this batch
                total_bytes = sum(
                    sum(s) if isinstance(s, list) else s for s in sizes
                )
                failed_codes = set(res[i] for i in failed)
                logger.warning(
                    "batch_put failed: %d/%d keys failed "
                    "(codes=%s, batch_bytes=%d, num_keys=%d), "
                    "first_key=%s",
                    len(failed),
                    len(keys),
                    failed_codes,
                    total_bytes,
                    len(keys),
                    keys[0] if keys else "N/A",
                )
                if (
                    MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                    and not self._mark_request_skipped_for_pressure(req_id)
                ):
                    logger.warning(
                        "Detected Mooncake CPU offloading pressure "
                        "(NO_AVAILABLE_HANDLE); skipping future store "
                        "batches for request %s until a later store "
                        "batch succeeds",
                        req_id,
                    )
            elif self._clear_store_pressure():
                logger.info(
                    "Mooncake CPU offloading pressure cleared after a "
                    "successful store batch"
                )
        except Exception as e:
            logger.error("Failed to put key %s, error: %s", keys, e)

        if self.enable_kv_event and stored_events:
            self.update_kv_event(stored_events)


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        store: Any,
        group_token_databases: list[ChunkedTokenDatabase]
        | ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        group_sw_blocks: list[int],
        ready_event: threading.Event,
    ):
        if isinstance(group_token_databases, ChunkedTokenDatabase):
            group_token_databases = [group_token_databases]
        super().__init__(
            store,
            group_token_databases[0]
            if group_token_databases
            else ChunkedTokenDatabase(
                KeyMetadata("", 0, 0, 0, 0), block_size
            ),
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )
        self.group_token_databases = group_token_databases
        self.group_sw_blocks = group_sw_blocks

    def _active_group_indices(
        self, block_ids: list[int] | tuple[list[int], ...]
    ) -> list[int]:
        if isinstance(block_ids, list):
            return [0]
        return list(range(len(block_ids)))

    def _handle_request(self, req_meta: ReqMeta):
        req_id = req_meta.req_id

        for group_idx in self._active_group_indices(req_meta.block_ids):
            token_database = self.group_token_databases[group_idx]
            if not token_database.kv_caches_base_addr:
                continue
            self._handle_request_for_group(req_meta, group_idx, token_database)

        self.set_finished_request(req_id)
        self.request_queue.task_done()

    def _handle_request_for_group(
        self,
        req_meta: ReqMeta,
        group_idx: int,
        token_database: ChunkedTokenDatabase,
    ):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // token_database.block_size
            * token_database.block_size
        )

        # Merge SWA mask so we only load blocks within the sliding window.
        sw_blocks = self.group_sw_blocks[group_idx]
        if sw_blocks > 0:
            total_blocks = (
                token_len + token_database.block_size - 1
            ) // token_database.block_size
            if total_blocks > sw_blocks:
                swa_mask = (total_blocks - sw_blocks) * token_database.block_size
                mask_num = max(mask_num, swa_mask)

        block_ids = _unwrap_block_ids(req_meta.block_ids, group_idx)

        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            addr, size, _ = token_database.prepare_value(
                start, end, block_ids
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)

        if not key_list:
            return

        # Rotate lists by tp_rank for load balancing
        key_list_c = (
            key_list[self.tp_rank % len(key_list) :]
            + key_list[: self.tp_rank % len(key_list)]
        )
        addr_list_c = (
            addr_list[self.tp_rank % len(addr_list) :]
            + addr_list[: self.tp_rank % len(addr_list)]
        )
        size_list_c = (
            size_list[self.tp_rank % len(size_list) :]
            + size_list[: self.tp_rank % len(size_list)]
        )

        try:
            res = self.store.batch_get_into_multi_buffers(
                key_list_c, addr_list_c, size_list_c
            )
            failed = [
                (key, value)
                for key, value in zip(key_list_c, res, strict=True)
                if value < 0
            ]
            if failed:
                logger.warning(
                    "Failed to get %d Mooncake keys (batch_keys=%d, first_failures=%s)",
                    len(failed),
                    len(key_list_c),
                    failed[:3],
                )
        except Exception as e:
            logger.warning(
                "Failed to get Mooncake batch %s, error: %s",
                key_list_c[:3],
                e,
            )


# ============================================================
# Store Worker
# ============================================================


class MooncakeStoreWorker:
    """Worker-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
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
        self.original_block_size = self.cache_config.block_size
        self.block_size = self.cache_config.block_size
        if self.pcp_size > 1:
            self.block_size *= self.pcp_size
        if self.dcp_size > 1:
            self.block_size *= self.dcp_size
        self.num_layers = model_config.get_num_layers(parallel_config)

        self.use_mla = False
        if (
            hasattr(model_config, "use_mla")
            and isinstance(model_config.use_mla, bool)
            and model_config.use_mla
        ):
            self.use_mla = True

        # Per-group parameters for HMA support.
        self._group_block_sizes: dict[int, int] = {}
        self._group_put_steps: dict[int, int] = {}
        self._group_head_or_tp_ranks: dict[int, int] = {}
        self._group_num_kv_heads: dict[int, int] = {}
        self._group_sw_blocks: dict[int, int] = {}
        if kv_cache_config is not None:
            self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
            self._layer_to_group: dict[str, int] = {}
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    self._layer_to_group[layer_name] = gid
                group_block_size = group.kv_cache_spec.block_size
                if self.pcp_size > 1:
                    group_block_size *= self.pcp_size
                if self.dcp_size > 1:
                    group_block_size *= self.dcp_size
                self._group_block_sizes[gid] = group_block_size

                # Compute per-group put_step, head_or_tp_rank and num_kv_head.
                spec = group.kv_cache_spec
                if hasattr(spec, "num_kv_heads"):
                    num_kv_head = spec.num_kv_heads
                else:
                    num_kv_head = 1
                self._group_num_kv_heads[gid] = num_kv_head
                if num_kv_head < self.tp_size:
                    put_step = self.tp_size // num_kv_head
                    head_or_tp_rank = self.tp_rank // put_step
                else:
                    head_or_tp_rank = self.tp_rank
                    put_step = 1
                self._group_put_steps[gid] = put_step
                self._group_head_or_tp_ranks[gid] = head_or_tp_rank

                # Compute per-group sliding window block count.
                if hasattr(spec, "sliding_window") and spec.sliding_window:
                    self._group_sw_blocks[gid] = (
                        cdiv(spec.sliding_window, group_block_size) + 1
                    )
                else:
                    self._group_sw_blocks[gid] = 0
        else:
            # Fallback for non-HMA: treat all layers as a single group.
            self.num_kv_cache_groups = 1
            self._layer_to_group = {}
            self._group_block_sizes[0] = self.block_size
            if self.use_mla:
                num_kv_head = 1
            else:
                num_kv_head = model_config.get_total_num_kv_heads()
            self._group_num_kv_heads[0] = num_kv_head
            if num_kv_head < self.tp_size:
                put_step = self.tp_size // num_kv_head
                head_or_tp_rank = self.tp_rank // put_step
            else:
                head_or_tp_rank = self.tp_rank
                put_step = 1
            self._group_put_steps[0] = put_step
            self._group_head_or_tp_ranks[0] = head_or_tp_rank
            self._group_sw_blocks[0] = 0

        # Default metadata (tp_rank=0); group-specific metadata is created
        # per-group in _register_group_caches.
        self.metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            tp_rank=0,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
            group_id=0,
        )

        # Per-group token databases (single entry for non-HMA)
        self.group_token_databases: list[ChunkedTokenDatabase] = []
        # Shared token_database for lookup (key generation is group-agnostic)
        self.token_database = ChunkedTokenDatabase(self.metadata, self.block_size)

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()

        local_seg = get_ip()
        config_dict = {
            "local_hostname": local_seg,
            "metadata_server": store_config.metadata_server,
            "global_segment_size": str(store_config.global_segment_size),
            "local_buffer_size": str(store_config.local_buffer_size),
            "protocol": store_config.protocol,
            "rdma_devices": store_config.device_name,
            "master_server_addr": store_config.master_server_address,
        }
        ret = self.store.setup(config_dict)
        if ret != 0:
            msg = "Initialize MooncakeDistributedStore failed."
            logger.error(msg)
            raise RuntimeError(msg)

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.finished_store_req: set[str] = set()

        # Start lookup server on rank 0 for scheduler-side prefix queries
        self.lookup_server: LookupKeyServer | None = None
        if vllm_config.parallel_config.rank == 0:
            self.lookup_server = LookupKeyServer(self, vllm_config)

    def _update_token_database_refs(self) -> None:
        """Update shared and backward-compatible references after registration.

        NOTE: We do NOT overwrite self.token_database here.  The lookup
        server uses process_tokens() which only needs the correct global
        block_size (self.block_size).  Overwriting with group 0's database
        would break lookup consistency if that group has a different block_size.
        """
        self.kv_caches_base_addr = []
        self.block_len = []
        for td in self.group_token_databases:
            self.kv_caches_base_addr.extend(td.kv_caches_base_addr)
            self.block_len.extend(td.block_len)

    def _start_transfer_threads(self) -> None:
        """Start KV cache store/load transfer threads."""
        if self.kv_send_thread is not None or self.kv_recv_thread is not None:
            return

        group_sw_blocks = [
            self._group_sw_blocks.get(gid, 0)
            for gid in range(self.num_kv_cache_groups)
        ]

        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            group_put_steps = [
                self._group_put_steps.get(gid, 1)
                for gid in range(self.num_kv_cache_groups)
            ]
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.group_token_databases,
                self.block_size,
                self.tp_rank,
                group_put_steps,
                group_sw_blocks,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
            )
            self.kv_send_thread.start()

        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.store,
            self.group_token_databases,
            self.block_size,
            self.tp_rank,
            group_sw_blocks,
            ready_event_recving,
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor.

        Wraps the unified tensor in a single-entry dict so that the
        existing stride-based logic in register_kv_caches() produces
        the correct single-segment result (block_len = page_size * num_layers).

        NOTE: HMA (multiple KV cache groups) is mutually exclusive with
        cross-layer blocks because use_uniform_kv_cache() requires exactly
        one attention group. This method should only be called in non-HMA
        mode, where register_kv_caches() correctly handles __cross_layer__.
        """
        assert not self._layer_to_group, (
            "register_cross_layers_kv_caches should not be called in HMA mode. "
            "use_uniform_kv_cache() prevents cross-layer when multiple groups exist."
        )
        self.register_kv_caches({"__cross_layer__": kv_cache})

    def _register_group_caches(
        self,
        group_caches: dict[str, torch.Tensor],
        group_id: int = 0,
        block_size: int | None = None,
        put_step: int | None = None,
        head_or_tp_rank: int | None = None,
    ) -> ChunkedTokenDatabase:
        """Register buffers for a single KV cache group and return its database.

        Args:
            group_caches: layer_name -> cache tensor for this group.
            block_size: The block size for this group. Defaults to self.block_size.
            put_step: The put_step for this group. Defaults to the group's
                put_step from _group_put_steps or 1.
            head_or_tp_rank: The effective TP rank for this group. Defaults to
                the group's value from _group_head_or_tp_ranks or self.tp_rank.
        """
        if block_size is None:
            block_size = self.block_size
        if put_step is None:
            put_step = 1
        if head_or_tp_rank is None:
            head_or_tp_rank = self.tp_rank

        first_kv_cache = next(iter(group_caches.values()))

        # num_blocks from cache_config is authoritative (set after
        # profiling, before KV cache allocation).
        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        # Detect the KV cache memory layout using the stride-based
        # approach from simple_kv_offload/worker.py.
        storage = first_kv_cache.untyped_storage()
        el = first_kv_cache.element_size()
        page_size_bytes = storage.nbytes() // self.num_blocks
        outer_dims = [
            d
            for d in range(first_kv_cache.ndim)
            if first_kv_cache.stride(d) * el > page_size_bytes
        ]

        # Register buffers with the store (deduplicate shared storages)
        # and record per-segment base addresses for every layer.
        seen_ptrs: set[int] = set()
        kv_caches_base_addr: list[int] = []
        block_len: list[int] = []

        for cache in group_caches.values():
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            region_len = cache_storage.nbytes()

            if base_addr not in seen_ptrs:
                seen_ptrs.add(base_addr)
                ret = self.store.register_buffer(base_addr, region_len)
                if ret != 0:
                    logger.error(
                        "register_buffer failed for addr %#x len %d: %d",
                        base_addr,
                        region_len,
                        ret,
                    )

            if not outer_dims:
                # Blocks-first layout (FlashInfer / MLA): one segment.
                kv_caches_base_addr.append(base_addr)
                block_len.append(page_size_bytes)
            else:
                # K/V-first layout (FlashAttn / ROCm): split segments.
                seg_stride = cache.stride(outer_dims[0]) * el
                for idx in range(cache.shape[outer_dims[0]]):
                    kv_caches_base_addr.append(base_addr + idx * seg_stride)
                    block_len.append(seg_stride // self.num_blocks)

        logger.info(
            "Registering KV_Caches. use_mla: %s, shape %s, "
            "num_blocks: %d, block_len: %s, "
            "per_key_bytes: %d, "
            "num_segments: %d",
            self.use_mla,
            first_kv_cache.shape,
            self.num_blocks,
            list(set(block_len)),
            sum(block_len),
            len(kv_caches_base_addr),
        )

        # Create group-specific metadata with the effective TP rank for keygen.
        group_metadata = KeyMetadata(
            model_name=self.metadata.model_name,
            tp_rank=head_or_tp_rank,
            pcp_rank=self.metadata.pcp_rank,
            dcp_rank=self.metadata.dcp_rank,
            pp_rank=self.metadata.pp_rank,
            group_id=group_id,
        )
        token_database = ChunkedTokenDatabase(group_metadata, block_size)
        token_database.set_kv_caches_base_addr(kv_caches_base_addr)
        token_database.set_block_len(block_len)
        return token_database

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors and start transfer threads."""
        # Group caches by their KV cache group ID for HMA support.
        group_caches: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        if self._layer_to_group:
            for layer_name, cache in kv_caches.items():
                gid = self._layer_to_group.get(layer_name)
                if gid is None:
                    continue
                group_caches[gid][layer_name] = cache
        else:
            # Non-HMA fallback: all layers go to group 0.
            group_caches[0] = dict(kv_caches)

        self.group_token_databases.clear()
        for gid in range(self.num_kv_cache_groups):
            caches = group_caches.get(gid, {})
            block_size = self._group_block_sizes.get(gid, self.block_size)
            put_step = self._group_put_steps.get(gid, 1)
            head_or_tp_rank = self._group_head_or_tp_ranks.get(
                gid, self.tp_rank
            )
            if not caches:
                # Create an empty database for groups without layers.
                group_metadata = KeyMetadata(
                    model_name=self.metadata.model_name,
                    tp_rank=head_or_tp_rank,
                    pcp_rank=self.metadata.pcp_rank,
                    dcp_rank=self.metadata.dcp_rank,
                    pp_rank=self.metadata.pp_rank,
                    group_id=gid,
                )
                self.group_token_databases.append(
                    ChunkedTokenDatabase(group_metadata, block_size)
                )
                continue
            token_database = self._register_group_caches(
                caches, gid, block_size, put_step, head_or_tp_rank
            )
            self.group_token_databases.append(token_database)

        self._update_token_database_refs()
        self._start_transfer_threads()

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

            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.block_size != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = load_spec.kvpool_cached_tokens + 1
            else:
                token_len = load_spec.kvpool_cached_tokens
            load_spec.token_len = token_len

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

    def _lookup_single_group(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        num_kv_head: int,
        token_database: ChunkedTokenDatabase,
    ) -> int:
        """Check prefix hit for a single group. Returns token count."""
        end = 0
        keys: list[str] = []
        try:
            starts: list[int] = []
            for start, end, key in token_database.process_tokens(
                token_len, block_hashes
            ):
                keys.append(key.to_string())
                starts.append(start)

            # Expand keys for all TP ranks
            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, num_kv_head)):
                for item in keys:
                    new_str = item.replace("@tp_rank:0", f"@tp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            # Expand keys for all PP ranks
            pp_base_keys = multi_tp_keys.copy()
            for i in range(1, self.pp_size):
                for item in pp_base_keys:
                    new_str = item.replace("@pp_rank:0", f"@pp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            res = self.store.batch_is_exist(multi_tp_keys)

            num_block = len(keys)
            multi_tp_values = [
                res[i * num_block : (i + 1) * num_block]
                for i in range(min(self.tp_size, num_kv_head) * self.pp_size)
            ]
            index = self._find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                return starts[index]
        except Exception as e:
            logger.error("Remote connection failed in lookup: %s", e)
            return 0
        return end

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
    ) -> int:
        """Check how many prefix tokens exist in the store.

        For non-HMA (single group) checks across all TP/PP ranks directly.
        For HMA, checks each group independently and returns the minimum
        prefix length found across all groups, ensuring every group has
        the required prefix blocks.
        """
        if self.num_kv_cache_groups <= 1:
            return self._lookup_single_group(
                token_len,
                block_hashes,
                self._group_num_kv_heads.get(0, 1),
                self.token_database,
            )

        # HMA: check each group independently and take the minimum.
        min_result = token_len
        for gid in range(self.num_kv_cache_groups):
            num_kv_head = self._group_num_kv_heads.get(gid, 1)
            group_block_size = self._group_block_sizes.get(gid, self.block_size)
            group_metadata = KeyMetadata(
                model_name=self.metadata.model_name,
                tp_rank=0,
                pcp_rank=self.pcp_rank,
                dcp_rank=self.dcp_rank,
                pp_rank=self.pp_rank,
                group_id=gid,
            )
            token_db = ChunkedTokenDatabase(group_metadata, group_block_size)
            result = self._lookup_single_group(
                token_len, block_hashes, num_kv_head, token_db
            )
            if result < min_result:
                min_result = result
            if min_result == 0:
                break
        return min_result

    @staticmethod
    def _find_min_first_non_one_index(
        arr: list[list[int]],
    ) -> int:
        try:
            return min(idx for row in arr for idx, val in enumerate(row) if val != 1)
        except ValueError:
            return -1

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []


# ============================================================
# Lookup Key Server
# ============================================================


class LookupKeyServer:
    """ZMQ server on worker rank 0 for handling prefix lookup queries."""

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
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                hash_frames = all_frames[1:]
                hashes_str = self.decoder.decode(hash_frames)
                result = self.store_worker.lookup(token_len, hashes_str)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

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
    """ZMQ client for querying prefix cache hits from worker."""

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
        all_frames = [token_len_bytes] + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    dp_rank = get_mooncake_dp_engine_index(vllm_config.parallel_config)
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    assert vllm_config.kv_transfer_config is not None
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    uid = os.getuid()
    logger.debug("Base URL: %s, RPC Port: %s, UID: %s", base_url, rpc_port, uid)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_uid{uid}_dp_rank{dp_rank}"
