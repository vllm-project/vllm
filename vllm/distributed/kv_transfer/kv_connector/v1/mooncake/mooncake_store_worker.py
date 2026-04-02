# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.serial_utils import MsgpackDecoder

from .mooncake_store_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from .mooncake_store_scheduler import get_zmq_rpc_path_lookup

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1 GiB


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
            protocol=config.get("protocol", "tcp"),
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
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.enable_kv_event = enable_kv_event

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

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        starts = []
        ends = []
        keys = []
        block_hashes: list[BlockHash] = []
        for index, (start, end, key) in enumerate(
            self.token_database.process_tokens(token_len, req_meta.block_hashes)
        ):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])

        # Apply put_step striding for TP
        starts = starts[self.tp_rank % self.put_step :: self.put_step]
        ends = ends[self.tp_rank % self.put_step :: self.put_step]
        keys = keys[self.tp_rank % self.put_step :: self.put_step]
        block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            return

        # Check which blocks already exist (dedup)
        exists_states = self.store.batch_is_exist(keys)
        missing_indices = [i for i, exists in enumerate(exists_states) if exists != 1]

        if not missing_indices:
            self.dec_stored_request(req_id)
            return

        starts = [starts[i] for i in missing_indices]
        ends = [ends[i] for i in missing_indices]
        keys = [keys[i] for i in missing_indices]
        block_hashes = [block_hashes[i] for i in missing_indices]

        logger.debug(
            "Storing KV cache for %d out of %d blocks "
            "(missing_count=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            len(missing_indices),
            req_id,
        )

        addrs = []
        sizes = []
        stored_events: list[BlockStored] = []
        prev_key = None
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

        for index, start in enumerate(starts):
            addr, size, _ = self.token_database.prepare_value(
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
            for value in res:
                if value < 0:
                    logger.error("Failed to put key %s, res: %s", keys, res)
        except Exception as e:
            logger.error("Failed to put key %s, error: %s", keys, e)

        if self.enable_kv_event and stored_events:
            self.update_kv_event(stored_events)

        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )

        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            addr, size, _ = self.token_database.prepare_value(
                start, end, req_meta.block_ids
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)

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
            for value in res:
                if value < 0:
                    logger.error(
                        "Failed to get key %s, res: %s",
                        key_list_c,
                        res,
                    )
        except Exception as e:
            logger.error("Failed to get key %s, error: %s", key_list_c, e)

        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
# Store Worker
# ============================================================


class MooncakeStoreWorker:
    """Worker-side component for MooncakeStoreConnector."""

    def __init__(self, vllm_config: VllmConfig):
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

        self.dp_rank = parallel_config.data_parallel_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", False
        )
        self.original_block_size = vllm_config.cache_config.block_size
        self.block_size = vllm_config.cache_config.block_size
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
        )

        self.token_database = ChunkedTokenDatabase(self.metadata, self.block_size)

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()

        local_seg = get_ip()
        ret = self.store.setup(
            local_seg,
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

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.finished_store_req: set[str] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors and start transfer threads."""
        # TODO(yifan): we haven't supported hybrid HMA yet.
        first_kv_cache = next(iter(kv_caches.values()))

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)

        # Per-block byte size: total tensor bytes / number of blocks.
        # Works for both MLA (3D: num_blocks, block_size, head_size) and
        # non-MLA (5D: num_blocks, 2, block_size, num_kv_heads, head_size).
        self.block_len: list[int] = [first_kv_cache.nbytes // self.num_blocks]

        logger.info(
            "Registering KV_Caches. use_mla: %s, shape %s",
            self.use_mla,
            first_kv_cache.shape,
        )

        self.kv_caches_base_addr: list[int] = []
        for cache in kv_caches.values():
            base_addr = cache.data_ptr()
            region_len = cache.nbytes
            self.kv_caches_base_addr.append(base_addr)
            self.store.register_buffer(base_addr, region_len)

        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)

        # Start transfer threads
        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
            )
            self.kv_send_thread.start()

        if self.load_async:
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreRecvingThread(
                self.store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                ready_event,
            )
            self.kv_recv_thread.start()
            ready_event.wait()

    def start_load_kv(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """Start loading KV cache from the store."""
        for request in metadata.requests:
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

            if self.load_async:
                assert self.kv_recv_thread is not None
                self.kv_recv_thread.add_request(request)
            else:
                # Synchronous load
                addr_list = []
                size_list = []
                key_list = []
                mask_num = (
                    load_spec.vllm_cached_tokens // self.block_size * self.block_size
                )
                for start, end, key in self.token_database.process_tokens(
                    token_len, request.block_hashes, mask_num
                ):
                    addr, size, _ = self.token_database.prepare_value(
                        start, end, request.block_ids
                    )
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)

                # Rotate by tp_rank for load balancing
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
                    self.store.batch_get_into_multi_buffers(
                        key_list_c, addr_list_c, size_list_c
                    )
                except Exception as e:
                    logger.error("Failed to get: %s", e)

    def wait_for_save(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """Record CUDA event and queue save requests."""
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
            self.kv_send_thread.add_stored_request(request.req_id)
            self.kv_send_thread.add_request(request)

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """Get completed send/recv request IDs."""
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

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
    ) -> int:
        """Check how many prefix tokens exist in the store.

        Checks across all TP ranks and PP ranks.
        """
        end = 0
        keys: list[str] = []
        try:
            starts: list[int] = []
            for start, end, key in self.token_database.process_tokens(
                token_len, block_hashes
            ):
                keys.append(key.to_string())
                starts.append(start)

            # Expand keys for all TP ranks
            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, self.num_kv_head)):
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
                for i in range(min(self.tp_size, self.num_kv_head) * self.pp_size)
            ]
            index = self._find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                return starts[index]
        except Exception as e:
            logger.error("Remote connection failed in lookup: %s", e)
            return 0
        return end

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
