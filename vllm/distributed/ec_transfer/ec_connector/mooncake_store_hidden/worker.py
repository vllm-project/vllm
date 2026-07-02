# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side hidden-state load/save logic for Mooncake Store."""

from __future__ import annotations

import os
import queue
import socket
import threading
import time

import torch
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HiddenKeyMetadata,
    HiddenPoolKey,
    HiddenSaveRequest,
    HiddenTensorDatabase,
    MMMeta,
    build_tensor_meta,
    validate_loaded_tensor,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.store_client import (
    HiddenStoreLoadError,
    MooncakeHiddenStoreClient,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

logger = init_logger(__name__)

LOOKUP_MSG = b"LOOKUP"
RESP_HIT = b"HIT"
RESP_MISS = b"MISS"
RESP_ERR = b"ERR"


class HiddenStoreWorker:
    """Synchronous hidden tensor load/save path used by the EC connector."""

    def __init__(
        self,
        store_client: MooncakeHiddenStoreClient,
        tensor_database: HiddenTensorDatabase | None = None,
        producer_engine_id: str | None = None,
        key_metadata: HiddenKeyMetadata | None = None,
    ):
        self.store_client = store_client
        self.tensor_database = tensor_database or HiddenTensorDatabase()
        self.producer_engine_id = producer_engine_id
        self.key_metadata = key_metadata
        self.sending_thread: HiddenStoreSendingThread | None = None

    def make_pool_key(self, identifier: str) -> HiddenPoolKey:
        assert self.key_metadata is not None
        return HiddenPoolKey(
            key_metadata=self.key_metadata,
            identifier=identifier,
        )

    def start_sending_thread(self) -> None:
        if self.sending_thread is not None:
            return
        self.sending_thread = HiddenStoreSendingThread(self)
        self.sending_thread.start()

    def enqueue_save(self, request: HiddenSaveRequest) -> None:
        if self.sending_thread is None:
            self.save_tensor(
                request.pool_key,
                request.tensor,
                now_ms=request.now_ms,
                with_soft_pin=request.with_soft_pin,
            )
            return
        self.sending_thread.add_request(request)

    def get_finished_sending(self) -> set[str]:
        if self.sending_thread is None:
            return set()
        return self.sending_thread.get_and_clear_finished_identifiers()

    def shutdown(self) -> None:
        if self.sending_thread is not None:
            self.sending_thread.close()
            self.sending_thread = None

    def lookup(self, identifier: str) -> bool:
        """Return whether the hidden object exists in Mooncake Store."""
        pool_key = self.make_pool_key(identifier)
        if not self.store_client.exists(pool_key):
            logger.info(
                "hidden_store_lookup_miss identifier=%s hidden_pool_key=%s "
                "reason=missing_object",
                pool_key.identifier,
                pool_key.to_string(),
            )
            return False

        logger.info(
            "hidden_store_lookup_hit identifier=%s hidden_pool_key=%s",
            pool_key.identifier,
            pool_key.to_string(),
        )
        return True

    def save_tensor(
        self,
        pool_key: HiddenPoolKey,
        tensor: torch.Tensor,
        now_ms: int | None = None,
        with_soft_pin: bool = False,
    ) -> None:
        if self.store_client.exists(pool_key):
            logger.debug(
                "hidden_store_save_skip identifier=%s hidden_pool_key=%s "
                "reason=exists",
                pool_key.identifier,
                pool_key.to_string(),
            )
            return

        started = time.perf_counter()
        stored_tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
        used_staging = stored_tensor is not tensor
        tensor_meta = build_tensor_meta(pool_key, stored_tensor)
        self.store_client.put_tensor(
            pool_key,
            stored_tensor,
            with_soft_pin=with_soft_pin,
        )
        logger.info(
            "hidden_store_put identifier=%s hidden_pool_key=%s nbytes=%d "
            "used_staging=%s hidden_store_put_ms=%.3f",
            pool_key.identifier,
            pool_key.to_string(),
            tensor_meta.nbytes,
            used_staging,
            (time.perf_counter() - started) * 1000.0,
        )

    def load(
        self,
        items: list[MMMeta],
        encoder_cache: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        for item in items:
            load_spec = item.load_spec
            if load_spec is None or not load_spec.can_load:
                continue
            if item.identifier in encoder_cache:
                logger.debug(
                    "hidden_store_load_skip identifier=%s "
                    "reason=local_encoder_cache",
                    item.identifier,
                )
                continue

            started = time.perf_counter()
            pool_key = self.make_pool_key(item.identifier)
            tensor_meta = self.store_client.get_tensor_meta(pool_key)
            if tensor_meta is None:
                raise HiddenStoreLoadError(
                    "failed to load hidden tensor metadata for "
                    f"{pool_key.to_string()}"
                )

            target_device = device
            if target_device is None:
                target_device = "cuda" if torch.cuda.is_available() else None
            target = torch.empty(
                tensor_meta.shape,
                dtype=_resolve_torch_dtype(tensor_meta.dtype),
                device=target_device,
            )
            _data_key, addrs, sizes = self.tensor_database.prepare_value(
                pool_key,
                target,
            )
            self.store_client.get_tensor_payload(
                pool_key,
                addrs[0],
                sizes[0],
                tensor_meta.data_offset,
            )
            validate_loaded_tensor(target, tensor_meta)
            encoder_cache[item.identifier] = target
            logger.info(
                "hidden_store_get identifier=%s hidden_pool_key=%s nbytes=%d "
                "hidden_store_get_ms=%.3f",
                item.identifier,
                pool_key.to_string(),
                tensor_meta.nbytes,
                (time.perf_counter() - started) * 1000.0,
            )


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "torch.float16":
        return torch.float16
    if dtype == "torch.bfloat16":
        return torch.bfloat16
    if dtype == "torch.float32":
        return torch.float32
    raise HiddenStoreLoadError(f"unsupported hidden tensor dtype: {dtype}")


class HiddenStoreSendingThread(threading.Thread):
    """Background thread for storing hidden tensors to the store."""

    def __init__(self, store_worker: HiddenStoreWorker):
        super().__init__(daemon=True, name="HiddenStoreSendingThread")
        self.store_worker = store_worker
        self.request_queue: queue.Queue[HiddenSaveRequest | None] = queue.Queue()
        self.done_task_lock = threading.Lock()
        self.finished_identifiers: set[str] = set()
        self._closed = threading.Event()

    def add_request(self, request: HiddenSaveRequest) -> None:
        self.request_queue.put(request)

    def get_and_clear_finished_identifiers(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_identifiers.copy()
            self.finished_identifiers.clear()
        return finished

    def set_finished_identifier(self, identifier: str) -> None:
        with self.done_task_lock:
            self.finished_identifiers.add(identifier)

    def run(self) -> None:
        while True:
            request = self.request_queue.get()
            try:
                if request is None:
                    return
                self.store_worker.save_tensor(
                    request.pool_key,
                    request.tensor,
                    now_ms=request.now_ms,
                    with_soft_pin=request.with_soft_pin,
                )
                self.set_finished_identifier(request.identifier)
            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)
            finally:
                self.request_queue.task_done()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.request_queue.put(None)


class HiddenLookupServer:
    """Worker rank-0 admin channel for scheduler-side hidden lookups."""

    def __init__(
        self,
        store_worker: HiddenStoreWorker,
        vllm_config: VllmConfig,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_hidden_lookup(vllm_config)
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
                    try:
                        identifier = bytes(all_frames[1]).decode("utf-8")
                        exists = self.store_worker.lookup(identifier)
                        if not exists:
                            self.socket.send_multipart([RESP_MISS])
                        else:
                            self.socket.send_multipart([RESP_HIT])
                    except Exception:
                        logger.exception("HiddenLookupServer lookup failed")
                        self.socket.send_multipart([RESP_ERR])
                else:
                    logger.warning(
                        "HiddenLookupServer received unknown msg_type: %r",
                        msg_type,
                    )
                    self.socket.send_multipart([RESP_ERR])

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        self.socket.close(linger=0)
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)


class HiddenLookupClient:
    """Scheduler-side client for worker rank-0 hidden lookup queries."""

    def __init__(self, vllm_config: VllmConfig):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_hidden_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, identifier: str) -> bool:
        self.socket.send_multipart([LOOKUP_MSG, identifier.encode("utf-8")])
        resp = self.socket.recv_multipart()
        msg_type = bytes(resp[0])
        if msg_type == RESP_HIT:
            return True
        if msg_type in (RESP_MISS, RESP_ERR):
            return False
        logger.warning("HiddenLookupClient received unknown response: %r", msg_type)
        return False

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_hidden_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for Hidden Store lookup socket."""
    assert vllm_config.ec_transfer_config is not None
    dp_rank = get_mooncake_dp_engine_index(vllm_config.parallel_config)
    base_url = envs.VLLM_RPC_BASE_PATH
    hostname = socket.gethostname()
    extra_config = vllm_config.ec_transfer_config.ec_connector_extra_config
    rpc_port = extra_config.get(
        "hidden_lookup_rpc_port",
        extra_config.get("lookup_rpc_port", 0),
    )
    logger.debug("Hidden lookup Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/hidden_lookup_rpc_port_{rpc_port}_host_{hostname}"
        f"_dp_rank{dp_rank}"
    )
