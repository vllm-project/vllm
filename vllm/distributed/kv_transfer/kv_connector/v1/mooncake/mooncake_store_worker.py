# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side logic for MooncakeStoreConnector.

Includes the store worker, transfer threads, lookup server,
and MooncakeDistributedStore integration.
"""

import ctypes
import json
import os
import queue
import threading
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
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
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (  # noqa: E501
    get_zmq_rpc_path_lookup,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.serial_utils import MsgpackDecoder

logger = init_logger(__name__)

DEFAULT_REQUESTER_LOCAL_BUFFER_SIZE = 1024 * 1024 * 1024  # 1 GiB
MOONCAKE_NO_AVAILABLE_HANDLE = -200
DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE = 1280 * 1024 * 1024
DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


class _DummyStagingPool:
    """Host-SHM staging ring for the dummy-client transfer path.

    The Mooncake dummy client rejects GPU pointers in register_buffer
    (`dummy_client.cpp:617`: "Dummy only register buffer within the shared
    memory region"). To offload from a GPU KV cache we stage every block
    through host SHM allocated from the dummy client's pool. The ring holds
    `num_slots` fixed-size slots; acquires block on a Condition when the ring
    is empty so callers get natural back-pressure instead of OOM.

    The pool is also pinned with cudaHostRegister so that cudaMemcpyAsync
    can pipeline transfers (pageable host forces a sync-per-call which is
    ~4x slower for the 32-segment KV blocks; see the benchmark in the
    commit message).
    """

    def __init__(self, store: Any, total_bytes: int, slot_bytes: int):
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be > 0, got {slot_bytes}")
        num_slots = total_bytes // slot_bytes
        if num_slots <= 0:
            raise ValueError(
                f"Dummy staging pool too small: total_bytes={total_bytes} < "
                f"slot_bytes={slot_bytes}. Increase 'local_buffer_size'."
            )
        usable = num_slots * slot_bytes
        self.base_addr: int = store.alloc_from_mem_pool(usable)
        if not self.base_addr:
            raise RuntimeError(
                f"alloc_from_mem_pool({usable}) returned 0; the dummy client's "
                f"local_buffer_size is likely too small"
            )
        ret = store.register_buffer(self.base_addr, usable)
        if ret != 0:
            raise RuntimeError(f"register_buffer for dummy staging pool failed: {ret}")
        self.slot_bytes = slot_bytes
        self.num_slots = num_slots
        self._total_bytes = usable
        self._free: list[int] = list(range(num_slots))
        self._cond = threading.Condition()

        # Pin the SHM so cudaMemcpyAsync can actually pipeline. A failure
        # here is non-fatal — we fall back to sync cudaMemcpy.
        self.pinned = False
        try:
            cudart = CudaRTLibrary()
            register = cudart.lib.cudaHostRegister
            register.restype = ctypes.c_int
            register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            CUDA_HOST_REGISTER_DEFAULT = 0
            rc = register(self.base_addr, usable, CUDA_HOST_REGISTER_DEFAULT)
            if rc != 0:
                logger.warning(
                    "cudaHostRegister(%#x, %d) failed: rc=%d -- falling "
                    "back to sync cudaMemcpy (slower)",
                    self.base_addr,
                    usable,
                    rc,
                )
            else:
                self.pinned = True
        except Exception as e:
            logger.warning(
                "Could not pin dummy staging pool: %s -- falling back to "
                "sync cudaMemcpy",
                e,
            )

    def slot_addr(self, slot: int) -> int:
        return self.base_addr + slot * self.slot_bytes

    def acquire(self, n: int) -> list[int]:
        if n > self.num_slots:
            raise RuntimeError(
                f"requested {n} staging slots but ring only holds "
                f"{self.num_slots}; increase 'local_buffer_size'"
            )
        with self._cond:
            while len(self._free) < n:
                self._cond.wait()
            taken = self._free[-n:]
            del self._free[-n:]
            return taken

    def release(self, slots: list[int]) -> None:
        if not slots:
            return
        with self._cond:
            self._free.extend(slots)
            self._cond.notify_all()


class _StagingCopier:
    """Thread-local staging memcpy issuer with three strategies.

    Strategies, picked at init by capability:

    * **batch** — single ``cuMemcpyBatchAsync`` per ``sync()`` covering every
      ``issue()`` triple. ~3-4x faster than per-segment async because the
      driver crossing is amortized. Requires the staging pool pinned and
      ``cuMemcpyBatchAsync`` available (CUDA 12.8+).
    * **async** — one ``cudaMemcpyAsync`` per ``issue()`` on a dedicated
      stream, single ``sync()`` per batch. Falls back here if the batch
      driver entrypoint isn't resolvable.
    * **sync** — plain ``cudaMemcpy`` per ``issue()`` (only used when the
      pool failed to pin; pageable host forces sync semantics anyway).
    """

    _CUDA_MEMCPY_DEFAULT = 4

    def __init__(self, cudart: CudaRTLibrary, pool_pinned: bool):
        self._cudart = cudart
        self._dsts: list[int] = []
        self._srcs: list[int] = []
        self._sizes: list[int] = []
        # Driver API calls (cuMemcpyBatchAsync) need a current CUDA context
        # on the calling thread. The send/recv threads have none until they
        # touch CUDA, so we lazily attach on first issue() from each thread.
        self._tls = threading.local()

        if not pool_pinned:
            self._mode = "sync"
            self._stream = None
            self._stream_handle = None
            self._memcpy_async = None
            self._batch_fn = None
            return

        self._stream = torch.cuda.Stream()
        self._stream_handle = self._stream.cuda_stream

        # Try the batch driver call first.
        try:
            from vllm.v1.simple_kv_offload.cuda_mem_ops import (
                _CUmemcpyAttributes,
                _resolve_batch_memcpy,
            )

            self._batch_fn = _resolve_batch_memcpy()
            self._attrs = _CUmemcpyAttributes(srcAccessOrder=3)  # ANY
            self._attrs_idx = ctypes.c_size_t(0)
            self._fail_idx = ctypes.c_size_t(0)
            self._mode = "batch"
            self._memcpy_async = None
            return
        except Exception as e:
            logger.info(
                "cuMemcpyBatchAsync unavailable (%s); falling back to "
                "per-segment cudaMemcpyAsync",
                e,
            )

        f = cudart.lib.cudaMemcpyAsync
        f.restype = ctypes.c_int
        f.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._memcpy_async = f
        self._batch_fn = None
        self._mode = "async"

    @property
    def mode(self) -> str:
        return self._mode

    def _ensure_context(self) -> None:
        """First call from a thread: attach the CUDA primary context.

        Driver-API calls (cuMemcpyBatchAsync, the cudaMemcpyAsync ctypes
        wrapper) require a current context on the calling thread. The
        spawned send/recv threads have none until they touch CUDA, so
        the very first driver call returns CUDA_ERROR_INVALID_VALUE.
        Allocating a one-element tensor implicitly attaches the primary
        context for the current accelerator on this thread.
        """
        if getattr(self._tls, "attached", False):
            return
        torch.zeros(1, device="cuda")
        self._tls.attached = True

    def issue(self, dst: int, src: int, size: int) -> None:
        if self._mode == "sync":
            self._cudart.cudaMemcpy(dst, src, size)
        elif self._mode == "async":
            self._ensure_context()
            assert self._memcpy_async is not None
            self._memcpy_async(
                dst, src, size, self._CUDA_MEMCPY_DEFAULT, self._stream_handle
            )
        else:  # batch
            self._dsts.append(dst)
            self._srcs.append(src)
            self._sizes.append(size)

    def sync(self) -> None:
        if self._mode == "sync":
            return
        if self._mode == "batch" and self._dsts:
            self._ensure_context()
            assert self._batch_fn is not None
            dsts = np.array(self._dsts, dtype=np.uint64)
            srcs = np.array(self._srcs, dtype=np.uint64)
            sizes = np.array(self._sizes, dtype=np.uint64)
            err = self._batch_fn(
                dsts.ctypes.data,
                srcs.ctypes.data,
                sizes.ctypes.data,
                len(self._dsts),
                ctypes.addressof(self._attrs),
                ctypes.byref(self._attrs_idx),
                1,
                ctypes.byref(self._fail_idx),
                self._stream_handle,
            )
            self._dsts.clear()
            self._srcs.clear()
            self._sizes.clear()
            if err != 0:
                raise RuntimeError(
                    f"cuMemcpyBatchAsync failed: err={err} "
                    f"fail_idx={self._fail_idx.value}"
                )
        # batch + async both need a stream sync
        assert self._stream is not None
        self._stream.synchronize()


@dataclass
class MooncakeStoreConfig:
    """Requester-facing configuration for MooncakeDistributedStore."""

    metadata_server: str
    requester_local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    # Dummy-client mode: vLLM colocates with a per-host Real Client process
    # (e.g. `mooncake_client`) that owns the RDMA resources, and this worker
    # forwards KV cache operations to it over IPC/RPC.
    enable_dummy_client: bool = False
    real_client_address: str = ""

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        enable_dummy_client = bool(
            _get_config_or_env_value(
                config,
                "enable_dummy_client",
                "MOONCAKE_ENABLE_DUMMY_CLIENT",
                False,
            )
        )
        real_client_address = _get_config_or_env_value(
            config,
            "real_client_address",
            "MOONCAKE_REAL_CLIENT_ADDRESS",
            "",
        )
        if enable_dummy_client and not real_client_address:
            raise ValueError(
                "'real_client_address' must be set when 'enable_dummy_client' is true."
            )
        return MooncakeStoreConfig(
            metadata_server=_get_config_or_env_value(
                config, "metadata_server", "MOONCAKE_TE_META_DATA_SERVER", ""
            ),
            requester_local_buffer_size=_get_requester_local_buffer_size(config),
            protocol=_get_config_or_env_value(
                config, "protocol", "MOONCAKE_PROTOCOL", "tcp"
            ),
            device_name=_get_config_or_env_value(
                config, "device_name", "MOONCAKE_DEVICE", ""
            ),
            master_server_address=_get_config_or_env_value(
                config, "master_server_address", "MOONCAKE_MASTER", ""
            ),
            enable_dummy_client=enable_dummy_client,
            real_client_address=real_client_address,
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


def _get_config_or_env_value(
    config: Mapping[str, Any], key: str, env_var: str, default: Any
) -> Any:
    env_value = os.getenv(env_var)
    if env_value not in (None, ""):
        return env_value
    return config.get(key, default)


def _get_requester_local_buffer_size(config: Mapping[str, Any]) -> int:
    value = config.get("local_buffer_size")
    if value is None:
        return DEFAULT_REQUESTER_LOCAL_BUFFER_SIZE
    return _parse_size(value)


def _get_kv_connector_extra_config(vllm_config: VllmConfig) -> Mapping[str, Any]:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return {}
    return kv_transfer_config.kv_connector_extra_config


def _get_disk_offload_buffer_budget_bytes() -> int:
    value = os.getenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES")
    if value is None:
        return DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE
    return _parse_size(value)


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


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    return max(1, int(raw_budget_bytes * DISK_OFFLOAD_USABLE_BUDGET_RATIO))


def _get_usable_disk_offload_batch_key_count(num_keys: int) -> int:
    return max(1, int(num_keys * DISK_OFFLOAD_USABLE_BUDGET_RATIO))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    max_batch_keys = _get_usable_disk_offload_batch_key_count(len(keys))
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
        if batch_keys and (
            batch_bytes + key_bytes > usable_budget_bytes
            or len(batch_keys) >= max_batch_keys
        ):
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
        replicate_config: Any | None = None,
        staging_pool: _DummyStagingPool | None = None,
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
        self.replicate_config = replicate_config
        self.staging_pool = staging_pool
        self._cudart: CudaRTLibrary | None = (
            CudaRTLibrary() if staging_pool is not None else None
        )
        self._copier: _StagingCopier | None = (
            _StagingCopier(self._cudart, staging_pool.pinned)
            if staging_pool is not None and self._cudart is not None
            else None
        )

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
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return
        if self._should_skip_request(req_id):
            logger.debug(
                "Skipping Mooncake store for request %s while CPU/disk offloading "
                "is under pressure",
                req_id,
            )
            self.dec_stored_request(req_id)
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

        slots: list[int] = []
        if self.staging_pool is not None:
            slots = self.staging_pool.acquire(len(keys))
            staged_addrs: list[list[int]] = []
            assert self._copier is not None
            for key_idx, (gpu_addrs_k, sizes_k) in enumerate(zip(addrs, sizes)):
                slot_base = self.staging_pool.slot_addr(slots[key_idx])
                sub_addrs: list[int] = []
                offset = 0
                for src_addr, sz in zip(gpu_addrs_k, sizes_k):
                    dst_addr = slot_base + offset
                    self._copier.issue(dst_addr, src_addr, sz)
                    sub_addrs.append(dst_addr)
                    offset += sz
                staged_addrs.append(sub_addrs)
            self._copier.sync()
            addrs = staged_addrs

        try:
            if self.replicate_config is None:
                res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes)
            else:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
            failed = [i for i, v in enumerate(res) if v < 0]
            if failed:
                # Compute total bytes attempted for this batch
                total_bytes = sum(sum(s) if isinstance(s, list) else s for s in sizes)
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
                        "Detected Mooncake CPU/disk offloading pressure "
                        "(NO_AVAILABLE_HANDLE); skipping future store "
                        "batches for request %s until a later store "
                        "batch succeeds",
                        req_id,
                    )
            elif self._clear_store_pressure():
                logger.info(
                    "Mooncake CPU/offload pressure cleared after a "
                    "successful store batch"
                )
        except Exception as e:
            logger.error("Failed to put key %s, error: %s", keys, e)
        finally:
            if slots and self.staging_pool is not None:
                self.staging_pool.release(slots)

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
        disk_offload_buffer_budget_bytes: int | None = None,
        staging_pool: _DummyStagingPool | None = None,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )
        self.staging_pool = staging_pool
        self._cudart: CudaRTLibrary | None = (
            CudaRTLibrary() if staging_pool is not None else None
        )
        self._copier: _StagingCopier | None = (
            _StagingCopier(self._cudart, staging_pool.pinned)
            if staging_pool is not None and self._cudart is not None
            else None
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

        load_batches = [(key_list_c, addr_list_c, size_list_c)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in size_list_c
            )
            usable_batch_keys = _get_usable_disk_offload_batch_key_count(
                len(key_list_c)
            )
            if (
                total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes
                or len(key_list_c) > usable_batch_keys
            ):
                assert self.disk_offload_buffer_budget_bytes is not None
                load_batches, oversized_key = _split_disk_offload_load_batches(
                    key_list_c,
                    addr_list_c,
                    size_list_c,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    oversized_key_index = key_list_c.index(oversized_key)
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

        current_batch_keys: list[str] = key_list_c
        try:
            for batch_keys, batch_addrs, batch_sizes in load_batches:
                current_batch_keys = batch_keys
                res = self._batch_get_into(batch_keys, batch_addrs, batch_sizes)
                failed = [
                    (key, value)
                    for key, value in zip(batch_keys, res, strict=True)
                    if value < 0
                ]
                if failed:
                    logger.warning(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        failed[:3],
                    )
                    break
        except Exception as e:
            logger.warning(
                "Failed to get Mooncake sub-batch %s, error: %s",
                current_batch_keys[:3],
                e,
            )

        self.set_finished_request(req_id)
        self.request_queue.task_done()

    def _batch_get_into(
        self,
        batch_keys: list[str],
        batch_addrs: list[list[int]],
        batch_sizes: list[list[int]],
    ) -> list[int]:
        """Call batch_get_into_multi_buffers, optionally staging via SHM.

        Real-client mode goes straight to the store with GPU buffers. Dummy
        mode lands bytes in SHM slots first, then copies SHM → GPU for keys
        that returned success (negative rc means the load failed; GPU memory
        for that key is left untouched so vLLM falls back to recompute).
        """
        if self.staging_pool is None:
            return self.store.batch_get_into_multi_buffers(
                batch_keys, batch_addrs, batch_sizes
            )

        assert self._copier is not None
        slots = self.staging_pool.acquire(len(batch_keys))
        try:
            staged_addrs: list[list[int]] = []
            for key_idx, sizes_k in enumerate(batch_sizes):
                slot_base = self.staging_pool.slot_addr(slots[key_idx])
                sub: list[int] = []
                offset = 0
                for sz in sizes_k:
                    sub.append(slot_base + offset)
                    offset += sz
                staged_addrs.append(sub)
            res = self.store.batch_get_into_multi_buffers(
                batch_keys, staged_addrs, batch_sizes
            )
            for key_idx, rc in enumerate(res):
                if rc < 0:
                    continue
                shm_addrs_k = staged_addrs[key_idx]
                gpu_addrs_k = batch_addrs[key_idx]
                sizes_k = batch_sizes[key_idx]
                for src_addr, dst_addr, sz in zip(shm_addrs_k, gpu_addrs_k, sizes_k):
                    self._copier.issue(dst_addr, src_addr, sz)
            self._copier.sync()
            return res
        finally:
            self.staging_pool.release(slots)


# ============================================================
# Store Worker
# ============================================================


class MooncakeStoreWorker:
    """Worker-side component for MooncakeStoreConnector."""

    def __init__(self, vllm_config: VllmConfig):
        try:
            from mooncake.store import (  # type: ignore  # noqa: E501
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

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        # NOTE(yifan): enforce load_async for now for better compute-I/O overlap.
        self.load_async = True
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
        extra_config = _get_kv_connector_extra_config(vllm_config)
        if not store_config.enable_dummy_client:
            store_config.device_name = rdma_utils.get_configured_worker_rnic(
                protocol=store_config.protocol,
                configured_device=store_config.device_name,
            )
        self.store = MooncakeDistributedStore()
        ret = self._setup_mooncake_store(self.store, store_config)
        if ret != 0:
            mode = "DummyClient" if store_config.enable_dummy_client else "RealClient"
            msg = f"Initialize MooncakeDistributedStore failed in {mode} mode."
            logger.error(msg)
            raise RuntimeError(msg)

        preferred_segment = rdma_utils.get_configured_preferred_segment(extra_config)
        self.preferred_segment = preferred_segment
        self.store_replicate_config = None
        if preferred_segment is not None:
            self.store_replicate_config = ReplicateConfig()
            self.store_replicate_config.preferred_segment = preferred_segment

        self.disk_offload_buffer_budget_bytes = _get_disk_offload_buffer_budget_bytes()

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.finished_store_req: set[str] = set()

        # In dummy-client mode, the wheel's register_buffer rejects raw GPU
        # pointers; we stage every block through host SHM. Pool is built once
        # the per-block byte size is known (in register_kv_caches).
        self.enable_dummy_client = store_config.enable_dummy_client
        self.dummy_local_buffer_size = store_config.requester_local_buffer_size
        self.dummy_staging_pool: _DummyStagingPool | None = None

    @staticmethod
    def _setup_mooncake_store(
        store: Any,
        store_config: MooncakeStoreConfig,
    ) -> int:
        if store_config.enable_dummy_client:
            logger.info(
                "Initializing MooncakeDistributedStore with DummyClient: "
                "real_client_address=%s, local_buffer_size=%d",
                store_config.real_client_address,
                store_config.requester_local_buffer_size,
            )
            return store.setup_dummy(
                0,
                store_config.requester_local_buffer_size,
                store_config.real_client_address,
            )

        local_ip = get_ip()
        local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
        logger.info(
            "Initializing MooncakeDistributedStore with RealClient: "
            "local_hostname=%s, metadata_server=%s, master=%s, "
            "protocol=%s, device_name=%s, local_buffer_size=%d",
            local_hostname,
            store_config.metadata_server,
            store_config.master_server_address,
            store_config.protocol,
            store_config.device_name,
            store_config.requester_local_buffer_size,
        )
        return store.setup(
            local_hostname,
            store_config.metadata_server,
            0,
            store_config.requester_local_buffer_size,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
        )

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor.

        Wraps the unified tensor in a single-entry dict so that the
        existing stride-based logic in register_kv_caches() produces
        the correct single-segment result (block_len = page_size * num_layers).
        """
        self.register_kv_caches({"__cross_layer__": kv_cache})

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors and start transfer threads."""
        # TODO(yifan): we haven't supported HMA yet.
        first_kv_cache = next(iter(kv_caches.values()))

        # num_blocks from cache_config is authoritative (set after
        # profiling, before KV cache allocation).
        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        # Detect the KV cache memory layout using the stride-based
        # approach from simple_kv_offload/worker.py.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm : (2, num_blocks, ...) → K/V outermost
        #   FlashInfer/MLA : (num_blocks, ...)    → blocks outermost
        #
        # We derive page_size_bytes = storage.nbytes() // num_blocks,
        # then classify dims: any dim whose byte-stride exceeds
        # page_size_bytes must be an outer segment dim (e.g. the K/V
        # dim of size 2).  For those backends we register each segment
        # (K, V) as a separate base-address so that the per-block
        # offset arithmetic in prepare_value() stays correct.
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
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []

        for cache in kv_caches.values():
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            region_len = cache_storage.nbytes()

            if base_addr not in seen_ptrs:
                seen_ptrs.add(base_addr)
                # Dummy mode rejects GPU pointers by design (see
                # _DummyStagingPool docstring); skip GPU registration and
                # rely on the SHM staging ring built below.
                if not getattr(self, "enable_dummy_client", False):
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
                self.kv_caches_base_addr.append(base_addr)
                self.block_len.append(page_size_bytes)
            else:
                # K/V-first layout (FlashAttn / ROCm): split segments.
                seg_stride = cache.stride(outer_dims[0]) * el
                for idx in range(cache.shape[outer_dims[0]]):
                    self.kv_caches_base_addr.append(base_addr + idx * seg_stride)
                    self.block_len.append(seg_stride // self.num_blocks)

        logger.info(
            "Registering KV_Caches. use_mla: %s, shape %s, "
            "num_blocks: %d, block_len: %s, "
            "per_key_bytes: %d, "
            "num_segments: %d",
            self.use_mla,
            first_kv_cache.shape,
            self.num_blocks,
            list(set(self.block_len)),
            sum(self.block_len),
            len(self.kv_caches_base_addr),
        )

        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)

        # Dummy-mode staging ring: one slot per in-flight block, slot size =
        # full per-block KV bytes (sum across segments).
        if getattr(self, "enable_dummy_client", False):
            slot_bytes = sum(self.block_len)
            self.dummy_staging_pool = _DummyStagingPool(
                self.store, self.dummy_local_buffer_size, slot_bytes
            )
            probe = _StagingCopier(CudaRTLibrary(), self.dummy_staging_pool.pinned)
            logger.info(
                "Dummy-client staging pool ready: base=%#x slots=%d "
                "slot_bytes=%d pinned=%s copier_mode=%s",
                self.dummy_staging_pool.base_addr,
                self.dummy_staging_pool.num_slots,
                slot_bytes,
                self.dummy_staging_pool.pinned,
                probe.mode,
            )

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
                getattr(self, "store_replicate_config", None),
                staging_pool=getattr(self, "dummy_staging_pool", None),
            )
            self.kv_send_thread.start()

        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.store,
            self.token_database,
            self.block_size,
            self.tp_rank,
            ready_event_recving,
            disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
            staging_pool=getattr(self, "dummy_staging_pool", None),
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
