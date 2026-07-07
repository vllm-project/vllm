# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side hidden-state load/save logic for Mooncake Store."""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HiddenKeyMetadata,
    HiddenPoolKey,
    HiddenSaveRequest,
    HiddenStoreOperationStats,
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
BATCH_LOOKUP_MSG = b"BATCH_LOOKUP"
RESP_BATCH = b"BATCH"
RESP_HIT = b"HIT"
RESP_MISS = b"MISS"
RESP_ERR = b"ERR"
THREAD_JOIN_TIMEOUT_SECONDS = 5.0


class HiddenStoreWorker:
    """Synchronous hidden tensor load/save path used by the EC connector."""

    def __init__(
        self,
        store_client: MooncakeHiddenStoreClient,
        tensor_database: HiddenTensorDatabase | None = None,
        key_metadata: HiddenKeyMetadata | None = None,
    ):
        self.store_client = store_client
        self.tensor_database = tensor_database or HiddenTensorDatabase()
        self.key_metadata = key_metadata
        self.sending_thread: HiddenStoreSendingThread | None = None
        self._operation_stats_lock = threading.Lock()
        self._operation_stats = HiddenStoreOperationStats()

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
                with_soft_pin=request.with_soft_pin,
            )
            return
        self.sending_thread.add_request(request)

    def get_finished_sending(self) -> set[str]:
        if self.sending_thread is None:
            return set()
        return self.sending_thread.get_and_clear_finished_identifiers()

    def get_failed_sending(self) -> dict[str, str]:
        if self.sending_thread is None:
            return {}
        return self.sending_thread.get_and_clear_failure_reasons()

    def get_operation_stats(self) -> HiddenStoreOperationStats | None:
        with self._operation_stats_lock:
            if self._operation_stats.is_empty():
                return None
            stats = self._operation_stats
            self._operation_stats = HiddenStoreOperationStats()
            return stats

    def _record_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._operation_stats_lock:
            self._operation_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def shutdown(self) -> None:
        if self.sending_thread is not None:
            self.sending_thread.close()
            self.sending_thread = None
        close_fn = getattr(self.store_client, "close", None)
        if close_fn is not None:
            close_fn()

    def lookup(self, identifier: str) -> bool:
        """Return whether the hidden object exists in Mooncake Store."""
        return self.lookup_batch([identifier]).get(identifier, False)

    def lookup_batch(self, identifiers: list[str]) -> dict[str, bool]:
        """Return whether hidden objects exist in Mooncake Store."""
        pool_keys = [self.make_pool_key(identifier) for identifier in identifiers]
        started = time.perf_counter()
        try:
            exists = self.store_client.batch_exists(pool_keys)
        except Exception:
            self._record_operation(
                "lookup_exists",
                time.perf_counter() - started,
                len(pool_keys),
                status="error",
                num_failed_keys=len(pool_keys),
            )
            raise

        failed_keys = sum(1 for hit in exists if not hit)
        self._record_operation(
            "lookup_exists",
            time.perf_counter() - started,
            len(pool_keys),
            status="miss" if failed_keys else "ok",
            num_failed_keys=failed_keys,
        )
        results = dict(zip(identifiers, exists, strict=True))
        for pool_key, hit in zip(pool_keys, exists, strict=True):
            if hit:
                logger.info(
                    "hidden_store_lookup_hit identifier=%s hidden_pool_key=%s",
                    pool_key.identifier,
                    pool_key.to_string(),
                )
            else:
                logger.info(
                    "hidden_store_lookup_miss identifier=%s hidden_pool_key=%s "
                    "reason=missing_object",
                    pool_key.identifier,
                    pool_key.to_string(),
                )
        return results

    def save_tensor(
        self,
        pool_key: HiddenPoolKey,
        tensor: torch.Tensor,
        with_soft_pin: bool = False,
    ) -> None:
        exists_started = time.perf_counter()
        try:
            exists = self.store_client.exists(pool_key)
        except Exception:
            self._record_operation(
                "save_exists",
                time.perf_counter() - exists_started,
                1,
                status="error",
                num_failed_keys=1,
            )
            raise

        self._record_operation(
            "save_exists",
            time.perf_counter() - exists_started,
            1,
            status="ok" if exists else "miss",
        )
        if exists:
            logger.info(
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
        try:
            self.store_client.put_tensor(
                pool_key,
                stored_tensor,
                with_soft_pin=with_soft_pin,
            )
        except Exception:
            self._record_operation(
                "save_put",
                time.perf_counter() - started,
                1,
                num_bytes=tensor_meta.nbytes,
                status="error",
                num_failed_keys=1,
            )
            raise
        self._record_operation(
            "save_put",
            time.perf_counter() - started,
            1,
            num_bytes=tensor_meta.nbytes,
            status="ok",
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
            tensor_meta = None
            load_stage = "metadata"
            try:
                tensor_meta = self.store_client.get_tensor_meta(pool_key)
                if tensor_meta is None:
                    raise HiddenStoreLoadError(
                        "failed to load hidden tensor metadata for "
                        f"{pool_key.to_string()}"
                    )

                load_stage = "allocate"
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
                load_stage = "payload"
                self.store_client.get_tensor_payload(
                    pool_key,
                    addrs[0],
                    sizes[0],
                    tensor_meta.data_offset,
                )
                load_stage = "validate"
                validate_loaded_tensor(target, tensor_meta)
            except Exception as e:
                self._record_operation(
                    "load_get",
                    time.perf_counter() - started,
                    1,
                    num_bytes=tensor_meta.nbytes if tensor_meta is not None else 0,
                    status="error",
                    num_failed_keys=1,
                )
                logger.exception(
                    "hidden_store_load_failed identifier=%s hidden_pool_key=%s "
                    "stage=%s shape=%s dtype=%s nbytes=%s error=%s",
                    item.identifier,
                    pool_key.to_string(),
                    load_stage,
                    tensor_meta.shape if tensor_meta is not None else None,
                    tensor_meta.dtype if tensor_meta is not None else None,
                    tensor_meta.nbytes if tensor_meta is not None else 0,
                    e,
                )
                raise
            encoder_cache[item.identifier] = target
            self._record_operation(
                "load_get",
                time.perf_counter() - started,
                1,
                num_bytes=tensor_meta.nbytes,
                status="ok",
            )
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
        self.failed_identifiers: set[str] = set()
        self.failure_reasons: dict[str, str] = {}
        self._closed = threading.Event()

    def add_request(self, request: HiddenSaveRequest) -> None:
        self.request_queue.put(request)

    def get_and_clear_finished_identifiers(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_identifiers.copy()
            self.finished_identifiers.clear()
        return finished

    def get_and_clear_failed_identifiers(self) -> set[str]:
        with self.done_task_lock:
            failed = self.failed_identifiers.copy()
            self.failed_identifiers.clear()
        return failed

    def get_and_clear_failure_reasons(self) -> dict[str, str]:
        with self.done_task_lock:
            failures = {
                identifier: self.failure_reasons.get(identifier, "")
                for identifier in self.failed_identifiers
            }
            for identifier in self.failed_identifiers:
                self.failure_reasons.pop(identifier, None)
            self.failed_identifiers.clear()
        return failures

    def set_finished_identifier(self, identifier: str) -> None:
        with self.done_task_lock:
            self.finished_identifiers.add(identifier)

    def set_failed_identifier(self, identifier: str, error: Exception) -> None:
        with self.done_task_lock:
            self.failed_identifiers.add(identifier)
            self.failure_reasons[identifier] = str(error)

    def run(self) -> None:
        while True:
            request = self.request_queue.get()
            try:
                if request is None:
                    return
                self.store_worker.save_tensor(
                    request.pool_key,
                    request.tensor,
                    with_soft_pin=request.with_soft_pin,
                )
                self.set_finished_identifier(request.identifier)
            except Exception as e:
                if request is not None:
                    self.set_failed_identifier(request.identifier, e)
                logger.error("Error in %s: %s", self.name, e)
            finally:
                self.request_queue.task_done()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.request_queue.put(None)
        if threading.current_thread() is not self:
            self.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)
            if self.is_alive():
                logger.warning(
                    "%s did not exit within %.1f seconds",
                    self.name,
                    THREAD_JOIN_TIMEOUT_SECONDS,
                )


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
                try:
                    all_frames = self.socket.recv_multipart(copy=False)
                except zmq.error.ZMQError:
                    if not self.running:
                        return
                    logger.exception("HiddenLookupServer recv failed")
                    continue
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
                elif msg_type == BATCH_LOOKUP_MSG:
                    try:
                        identifiers = [
                            bytes(frame).decode("utf-8") for frame in all_frames[1:]
                        ]
                        exists = self.store_worker.lookup_batch(identifiers)
                        frames = [
                            RESP_HIT if exists.get(identifier, False) else RESP_MISS
                            for identifier in identifiers
                        ]
                        self.socket.send_multipart([RESP_BATCH, *frames])
                    except Exception:
                        logger.exception("HiddenLookupServer batch lookup failed")
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
        self.thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            logger.warning(
                "HiddenLookupServer thread did not exit within %.1f seconds",
                THREAD_JOIN_TIMEOUT_SECONDS,
            )
        _close_zmq_context(self.ctx)
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
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="HiddenLookupClient",
        )
        self.futures: dict[str, Future[dict[str, bool]]] = {}

    def lookup(self, identifier: str) -> bool:
        result = self.lookup_batch([identifier], non_block=False)
        assert result is not None
        return result.get(identifier, False)

    def _lookup_batch(self, identifiers: list[str]) -> dict[str, bool]:
        self.socket.send_multipart(
            [
                BATCH_LOOKUP_MSG,
                *(identifier.encode("utf-8") for identifier in identifiers),
            ]
        )
        resp = self.socket.recv_multipart()
        msg_type = bytes(resp[0])
        if msg_type == RESP_BATCH:
            states = [bytes(frame) == RESP_HIT for frame in resp[1:]]
            if len(states) != len(identifiers):
                logger.warning(
                    "HiddenLookupClient received malformed batch response: "
                    "identifiers=%d states=%d",
                    len(identifiers),
                    len(states),
                )
                return {identifier: False for identifier in identifiers}
            return dict(zip(identifiers, states, strict=True))
        if msg_type == RESP_ERR:
            return {identifier: False for identifier in identifiers}
        logger.warning("HiddenLookupClient received unknown response: %r", msg_type)
        return {identifier: False for identifier in identifiers}

    def lookup_batch(
        self,
        identifiers: list[str],
        non_block: bool = False,
    ) -> dict[str, bool] | None:
        identifiers = list(dict.fromkeys(identifiers))
        if not identifiers:
            return {}

        new_identifiers = [
            identifier for identifier in identifiers if identifier not in self.futures
        ]
        if new_identifiers:
            future = self.executor.submit(self._lookup_batch, new_identifiers)
            for identifier in new_identifiers:
                self.futures[identifier] = future

        if non_block and any(
            not self.futures[identifier].done() for identifier in identifiers
        ):
            return None

        results: dict[str, bool] = {}
        for identifier in identifiers:
            future = self.futures[identifier]
            try:
                batch_results = future.result()
                results[identifier] = batch_results.get(identifier, False)
            except Exception as e:
                logger.error("Async hidden lookup failed for %s: %s", identifier, e)
                results[identifier] = False
            finally:
                self.futures.pop(identifier, None)
        return results

    def discard(self, identifier: str) -> None:
        future = self.futures.pop(identifier, None)
        if future is None:
            return
        if not any(existing is future for existing in self.futures.values()):
            future.cancel()

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.futures.clear()
        self.socket.close(linger=0)
        _close_zmq_context(self.ctx)


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


def _close_zmq_context(ctx) -> None:
    try:
        destroy = getattr(ctx, "destroy", None)
        if destroy is not None:
            destroy(linger=0)
            return
        term = getattr(ctx, "term", None)
        if term is not None:
            term()
    except Exception:
        logger.warning("failed to close hidden lookup ZMQ context", exc_info=True)
