# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FS lookup subprocess transport using vLLM MessageQueue."""

import contextlib
import os
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, cast

from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.system_utils import get_mp_context
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import (
    BaseAsyncLookupManager,
    LookupBatch,
    LookupResults,
)

if TYPE_CHECKING:
    import multiprocessing

logger = init_logger(__name__)

_READY_TIMEOUT_S = 30.0


class FsIoProcess:
    """Owns the filesystem lookup subprocess and MessageQueue channels."""

    def __init__(self, tier_type: str, file_mapper: FileMapper):
        ctx = get_mp_context()
        ready_reader, ready_writer = ctx.Pipe(duplex=False)
        self._ready_reader: Connection = ready_reader

        # Scheduler writer -> process reader.
        self._request_queue = MessageQueue(n_reader=1, n_local_reader=1)
        self._result_queue: MessageQueue | None = None
        self._shutting_down = False

        self._process: BaseProcess = ctx.Process(
            target=_fs_lookup_worker_main,
            kwargs={
                "tier_type": tier_type,
                "file_mapper": file_mapper,
                "request_handle": self._request_queue.export_handle(),
                "ready_pipe": ready_writer,
            },
            name=f"vllm_offloading_lookup_{tier_type}",
            daemon=True,
        )
        self._process.start()
        ready_writer.close()

        try:
            self._wait_for_startup_handshake()
        except Exception:
            self.shutdown()
            raise

    def _wait_for_startup_handshake(self) -> None:
        if not self._ready_reader.poll(_READY_TIMEOUT_S):
            raise RuntimeError(
                "Timed out waiting for fs lookup process result queue handle"
            )

        message = self._ready_reader.recv()
        if not isinstance(message, dict) or "result_handle" not in message:
            raise RuntimeError(f"Unexpected fs lookup startup message: {message!r}")
        result_handle = cast(Handle, message["result_handle"])
        self._result_queue = MessageQueue.create_from_handle(result_handle, rank=0)

        # MessageQueue readiness is collective: parent and child must both call.
        self._request_queue.wait_until_ready()
        self._result_queue.wait_until_ready()

        if not self._ready_reader.poll(_READY_TIMEOUT_S):
            raise RuntimeError("Timed out waiting for fs lookup process ready signal")
        ready_signal = self._ready_reader.recv()
        if ready_signal != "READY":
            raise RuntimeError(f"Unexpected fs lookup ready signal: {ready_signal!r}")

    def enqueue_lookup_batch(self, batch: LookupBatch) -> None:
        self._request_queue.enqueue(batch)

    def dequeue_lookup_results_nowait(self) -> LookupResults | None:
        assert self._result_queue is not None
        try:
            return cast(LookupResults, self._result_queue.dequeue(timeout=0))
        except TimeoutError:
            return None

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        with contextlib.suppress(Exception):
            self._request_queue.enqueue(None)

        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)

        if self._result_queue is not None:
            self._result_queue.shutdown()
        self._request_queue.shutdown()
        self._ready_reader.close()


def _fs_lookup_worker_main(
    tier_type: str,
    file_mapper: FileMapper,
    request_handle: Handle,
    ready_pipe: "multiprocessing.connection.Connection",
) -> None:
    request_reader = MessageQueue.create_from_handle(request_handle, rank=0)
    result_writer = MessageQueue(n_reader=1, n_local_reader=1)
    try:
        ready_pipe.send({"result_handle": result_writer.export_handle()})

        request_reader.wait_until_ready()
        result_writer.wait_until_ready()
        ready_pipe.send("READY")
        ready_pipe.close()

        def _batch_lookup(
            keys: list[OffloadKey], req_context: ReqContext
        ) -> list[bool]:
            del req_context
            return [os.path.exists(file_mapper.get_file_name(key)) for key in keys]

        while True:
            pending = cast(LookupBatch | None, request_reader.dequeue(indefinite=True))
            if pending is None:
                break
            results = BaseAsyncLookupManager.execute_lookup_batch(
                pending,
                tier_type,
                _batch_lookup,
            )
            if results:
                result_writer.enqueue(results)
    finally:
        request_reader.shutdown()
        result_writer.shutdown()
        with contextlib.suppress(Exception):
            ready_pipe.close()
