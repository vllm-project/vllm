# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side embedding load/save logic for Mooncake Store."""

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
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    EmbeddingKeyMetadata,
    EmbeddingPoolKey,
    EmbeddingSaveRequest,
    EmbeddingStoreOperationStats,
    EmbeddingTensorDatabase,
    MMMeta,
    build_tensor_meta,
    validate_loaded_tensor,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

from .store_client import (
    EmbeddingStoreLoadError,
    EmbeddingStoreSaveError,
    MooncakeEmbeddingStoreClient,
)

logger = init_logger(__name__)

LOOKUP_MSG = b"LOOKUP"
BATCH_LOOKUP_MSG = b"BATCH_LOOKUP"
RESP_BATCH = b"BATCH"
RESP_HIT = b"HIT"
RESP_MISS = b"MISS"
RESP_ERR = b"ERR"
THREAD_JOIN_TIMEOUT_SECONDS = 5.0


class EmbeddingStoreWorker:
    """Synchronous embedding tensor load/save path used by the EC connector."""

    def __init__(
        self,
        store_client: MooncakeEmbeddingStoreClient,
        tensor_database: EmbeddingTensorDatabase | None = None,
        key_metadata: EmbeddingKeyMetadata | None = None,
    ):
        self.store_client = store_client
        self.tensor_database = tensor_database or EmbeddingTensorDatabase()
        self.key_metadata = key_metadata
        self.sending_thread: EmbeddingStoreSendingThread | None = None
        self._operation_stats_lock = threading.Lock()
        self._operation_stats = EmbeddingStoreOperationStats()

    def make_pool_key(self, identifier: str) -> EmbeddingPoolKey:
        assert self.key_metadata is not None
        return EmbeddingPoolKey(
            key_metadata=self.key_metadata,
            identifier=identifier,
        )

    def start_sending_thread(self) -> None:
        if self.sending_thread is not None:
            return
        self.sending_thread = EmbeddingStoreSendingThread(self)
        self.sending_thread.start()

    def enqueue_save(self, request: EmbeddingSaveRequest) -> None:
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

    def wait_for_pending_saves(self) -> None:
        if self.sending_thread is not None:
            self.sending_thread.request_queue.join()

    def get_operation_stats(self) -> EmbeddingStoreOperationStats | None:
        with self._operation_stats_lock:
            if self._operation_stats.is_empty():
                return None
            stats = self._operation_stats
            self._operation_stats = EmbeddingStoreOperationStats()
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
        """Return whether the embedding object exists in Mooncake Store."""
        return self.lookup_batch([identifier]).get(identifier, False)

    def lookup_batch(self, identifiers: list[str]) -> dict[str, bool]:
        """Return whether embedding objects exist in Mooncake Store."""
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
                    "embedding_store_lookup_hit identifier=%s embedding_pool_key=%s",
                    pool_key.identifier,
                    pool_key.to_string(),
                )
            else:
                logger.info(
                    "embedding_store_lookup_miss identifier=%s embedding_pool_key=%s "
                    "reason=missing_object",
                    pool_key.identifier,
                    pool_key.to_string(),
                )
        return results

    def save_tensor(
        self,
        pool_key: EmbeddingPoolKey,
        tensor: torch.Tensor,
        with_soft_pin: bool = False,
    ) -> None:
        request = EmbeddingSaveRequest(
            pool_key=pool_key,
            tensor=tensor,
            with_soft_pin=with_soft_pin,
        )
        error = self.save_tensors([request])[0]
        if error is not None:
            raise error

    def save_tensors(
        self,
        requests: list[EmbeddingSaveRequest],
    ) -> list[Exception | None]:
        """Store a batch and return one error slot per request."""
        if not requests:
            return []

        errors: list[Exception | None] = [None] * len(requests)
        pool_keys = [request.pool_key for request in requests]
        exists_started = time.perf_counter()
        try:
            exists = self.store_client.batch_exists(pool_keys)
            if len(exists) != len(requests):
                raise EmbeddingStoreSaveError(
                    "Mooncake batch exists returned an unexpected number of results: "
                    f"expected {len(requests)}, got {len(exists)}"
                )
        except Exception as error:
            self._record_operation(
                "save_exists",
                time.perf_counter() - exists_started,
                len(requests),
                status="error",
                num_failed_keys=len(requests),
            )
            return [error] * len(requests)

        missing_count = sum(not hit for hit in exists)
        self._record_operation(
            "save_exists",
            time.perf_counter() - exists_started,
            len(requests),
            status="miss" if missing_count else "ok",
        )
        missing_by_soft_pin: dict[bool, list[int]] = {False: [], True: []}
        stored_tensors: dict[int, torch.Tensor] = {}
        tensor_nbytes: dict[int, int] = {}
        used_staging: dict[int, bool] = {}
        for index, (request, hit) in enumerate(zip(requests, exists, strict=True)):
            if hit:
                logger.info(
                    "embedding_store_save_skip identifier=%s embedding_pool_key=%s "
                    "reason=exists",
                    request.identifier,
                    request.pool_key.to_string(),
                )
                continue
            tensor = request.tensor
            stored_tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
            stored_tensors[index] = stored_tensor
            used_staging[index] = stored_tensor is not tensor
            tensor_nbytes[index] = build_tensor_meta(
                request.pool_key, stored_tensor
            ).nbytes
            staging_event = _record_tensor_ready_event(stored_tensor)
            _wait_tensor_ready_event(staging_event)
            missing_by_soft_pin[request.with_soft_pin].append(index)

        for with_soft_pin, indices in missing_by_soft_pin.items():
            if not indices:
                continue
            started = time.perf_counter()
            total_nbytes = sum(tensor_nbytes[index] for index in indices)
            try:
                results = self.store_client.put_tensors(
                    [requests[index].pool_key for index in indices],
                    [stored_tensors[index] for index in indices],
                    with_soft_pin=with_soft_pin,
                )
            except Exception as error:
                results = [False] * len(indices)
                for index in indices:
                    errors[index] = error
            for index, success in zip(indices, results, strict=True):
                if not success and errors[index] is None:
                    errors[index] = EmbeddingStoreSaveError(
                        "failed to put embedding tensor for "
                        f"{requests[index].pool_key.to_string()}"
                    )

            num_failed = sum(not success for success in results)
            self._record_operation(
                "save_put",
                time.perf_counter() - started,
                len(indices),
                num_bytes=total_nbytes,
                status="error" if num_failed else "ok",
                num_failed_keys=num_failed,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "embedding_store_batch_put num_keys=%d num_failed_keys=%d "
                "nbytes=%d embedding_store_put_ms=%.3f",
                len(indices),
                num_failed,
                total_nbytes,
                elapsed_ms,
            )
            for index, success in zip(indices, results, strict=True):
                if success:
                    request = requests[index]
                    logger.info(
                        "embedding_store_put identifier=%s embedding_pool_key=%s "
                        "nbytes=%d used_staging=%s embedding_store_put_ms=%.3f",
                        request.identifier,
                        request.pool_key.to_string(),
                        tensor_nbytes[index],
                        used_staging[index],
                        elapsed_ms,
                    )
        return errors

    def load(
        self,
        items: list[MMMeta],
        encoder_cache: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        load_items: list[MMMeta] = []
        for item in items:
            load_spec = item.load_spec
            if load_spec is None or not load_spec.can_load:
                continue
            if item.identifier in encoder_cache:
                logger.debug(
                    "embedding_store_load_skip identifier=%s "
                    "reason=local_encoder_cache",
                    item.identifier,
                )
                continue
            load_items.append(item)
        if not load_items:
            return

        pool_keys = [self.make_pool_key(item.identifier) for item in load_items]
        started = time.perf_counter()
        try:
            tensor_metas = self.store_client.get_tensor_metas(pool_keys)
            for index, tensor_meta in enumerate(tensor_metas):
                if tensor_meta is None:
                    raise EmbeddingStoreLoadError(
                        "failed to load embedding tensor metadata for "
                        f"{pool_keys[index].to_string()}"
                    )
        except Exception as e:
            self._record_operation(
                "load_get",
                time.perf_counter() - started,
                len(load_items),
                status="error",
                num_failed_keys=len(load_items),
            )
            logger.exception(
                "embedding_store_load_failed stage=metadata num_items=%d error=%s",
                len(load_items),
                e,
            )
            raise

        target_device = device
        if target_device is None:
            target_device = "cuda" if torch.cuda.is_available() else None
        targets = [
            torch.empty(
                tensor_meta.shape,
                dtype=_resolve_torch_dtype(tensor_meta.dtype),
                device=target_device,
            )
            for tensor_meta in tensor_metas
        ]
        addrs: list[int] = []
        sizes: list[int] = []
        for pool_key, target in zip(pool_keys, targets, strict=True):
            _data_key, item_addrs, item_sizes = self.tensor_database.prepare_value(
                pool_key,
                target,
            )
            addrs.extend(item_addrs)
            sizes.extend(item_sizes)
        data_offsets = [tensor_meta.data_offset for tensor_meta in tensor_metas]

        try:
            self.store_client.get_tensor_payloads(
                pool_keys,
                addrs,
                sizes,
                data_offsets,
            )
        except Exception as e:
            self._record_operation(
                "load_get",
                time.perf_counter() - started,
                len(load_items),
                num_bytes=sum(meta.nbytes for meta in tensor_metas),
                status="error",
                num_failed_keys=len(load_items),
            )
            logger.exception(
                "embedding_store_load_failed stage=payload num_items=%d error=%s",
                len(load_items),
                e,
            )
            raise

        for item, tensor_meta, target, pool_key in zip(
            load_items,
            tensor_metas,
            targets,
            pool_keys,
            strict=True,
        ):
            try:
                validate_loaded_tensor(target, tensor_meta)
            except Exception as e:
                self._record_operation(
                    "load_get",
                    time.perf_counter() - started,
                    1,
                    num_bytes=tensor_meta.nbytes,
                    status="error",
                    num_failed_keys=1,
                )
                logger.exception(
                    "embedding_store_load_failed identifier=%s "
                    "embedding_pool_key=%s stage=validate shape=%s dtype=%s "
                    "nbytes=%s error=%s",
                    item.identifier,
                    pool_key.to_string(),
                    tensor_meta.shape,
                    tensor_meta.dtype,
                    tensor_meta.nbytes,
                    e,
                )
                raise
            encoder_cache[item.identifier] = target
            logger.info(
                "embedding_store_get identifier=%s embedding_pool_key=%s nbytes=%d "
                "embedding_store_get_ms=%.3f",
                item.identifier,
                pool_key.to_string(),
                tensor_meta.nbytes,
                (time.perf_counter() - started) * 1000.0,
            )

        self._record_operation(
            "load_get",
            time.perf_counter() - started,
            len(load_items),
            num_bytes=sum(tensor_meta.nbytes for tensor_meta in tensor_metas),
            status="ok",
        )


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "torch.float16":
        return torch.float16
    if dtype == "torch.bfloat16":
        return torch.bfloat16
    if dtype == "torch.float32":
        return torch.float32
    raise EmbeddingStoreLoadError(f"unsupported embedding tensor dtype: {dtype}")


def _record_tensor_ready_event(tensor: torch.Tensor) -> torch.Event | None:
    if not tensor.is_cuda:
        return None
    event = torch.Event()
    event.record()
    return event


def _wait_tensor_ready_event(event: torch.Event | None) -> None:
    if event is not None:
        event.synchronize()


class EmbeddingStoreSendingThread(threading.Thread):
    """Background thread for storing embedding tensors to the store."""

    def __init__(self, store_worker: EmbeddingStoreWorker):
        super().__init__(daemon=True, name="EmbeddingStoreSendingThread")
        self.store_worker = store_worker
        self.request_queue: queue.Queue[EmbeddingSaveRequest | None] = queue.Queue()
        self.done_task_lock = threading.Lock()
        self.finished_identifiers: set[str] = set()
        self.failed_identifiers: set[str] = set()
        self.failure_reasons: dict[str, str] = {}
        self._closed = threading.Event()

    def add_request(self, request: EmbeddingSaveRequest) -> None:
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
            first_request = self.request_queue.get()
            if first_request is None:
                self.request_queue.task_done()
                return

            requests = [first_request]
            should_close = False
            while True:
                try:
                    request = self.request_queue.get_nowait()
                except queue.Empty:
                    break
                if request is None:
                    should_close = True
                    self.request_queue.task_done()
                    break
                requests.append(request)

            ready_requests = []
            try:
                for request in requests:
                    try:
                        _wait_tensor_ready_event(request.ready_event)
                    except Exception as event_error:
                        self.set_failed_identifier(request.identifier, event_error)
                        logger.error("Error in %s: %s", self.name, event_error)
                    else:
                        ready_requests.append(request)

                errors = self.store_worker.save_tensors(ready_requests)
                for request, save_error in zip(ready_requests, errors, strict=True):
                    if save_error is None:
                        self.set_finished_identifier(request.identifier)
                    else:
                        self.set_failed_identifier(request.identifier, save_error)
                        logger.error("Error in %s: %s", self.name, save_error)
            except Exception as batch_error:
                for request in ready_requests:
                    self.set_failed_identifier(request.identifier, batch_error)
                logger.error("Error in %s: %s", self.name, batch_error)
            finally:
                for _ in requests:
                    self.request_queue.task_done()

            if should_close:
                return

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


class EmbeddingLookupServer:
    """Worker rank-0 admin channel for scheduler-side embedding lookups."""

    def __init__(
        self,
        store_worker: EmbeddingStoreWorker,
        vllm_config: VllmConfig,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_embedding_lookup(vllm_config)
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
                    logger.exception("EmbeddingLookupServer recv failed")
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
                        logger.exception("EmbeddingLookupServer lookup failed")
                        self.socket.send_multipart([RESP_ERR])
                elif msg_type == BATCH_LOOKUP_MSG:
                    try:
                        identifiers = [
                            bytes(frame).decode("utf-8") for frame in all_frames[1:]
                        ]
                        exists_by_identifier = self.store_worker.lookup_batch(
                            identifiers
                        )
                        frames = [
                            RESP_HIT
                            if exists_by_identifier.get(identifier, False)
                            else RESP_MISS
                            for identifier in identifiers
                        ]
                        self.socket.send_multipart([RESP_BATCH, *frames])
                    except Exception:
                        logger.exception("EmbeddingLookupServer batch lookup failed")
                        self.socket.send_multipart([RESP_ERR])
                else:
                    logger.warning(
                        "EmbeddingLookupServer received unknown msg_type: %r",
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
                "EmbeddingLookupServer thread did not exit within %.1f seconds",
                THREAD_JOIN_TIMEOUT_SECONDS,
            )
        _close_zmq_context(self.ctx)
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)


class EmbeddingLookupClient:
    """Scheduler-side client for worker rank-0 embedding lookup queries."""

    def __init__(self, vllm_config: VllmConfig):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_embedding_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="EmbeddingLookupClient",
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
                    "EmbeddingLookupClient received malformed batch response: "
                    "identifiers=%d states=%d",
                    len(identifiers),
                    len(states),
                )
                return {identifier: False for identifier in identifiers}
            return dict(zip(identifiers, states, strict=True))
        if msg_type == RESP_ERR:
            return {identifier: False for identifier in identifiers}
        logger.warning("EmbeddingLookupClient received unknown response: %r", msg_type)
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
                logger.error("Async embedding lookup failed for %s: %s", identifier, e)
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


def get_zmq_rpc_path_embedding_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for Embedding Store lookup socket."""
    assert vllm_config.ec_transfer_config is not None
    dp_rank = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    hostname = socket.gethostname()
    extra_config = vllm_config.ec_transfer_config.ec_connector_extra_config
    rpc_port = extra_config.get(
        "embedding_lookup_rpc_port",
        extra_config.get("lookup_rpc_port", 0),
    )
    logger.debug("Embedding lookup Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/embedding_lookup_rpc_port_{rpc_port}_host_{hostname}"
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
        logger.warning("failed to close embedding lookup ZMQ context", exc_info=True)
