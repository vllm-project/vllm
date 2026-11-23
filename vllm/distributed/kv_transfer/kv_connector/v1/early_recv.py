# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report completed KV loads ahead of the next model output."""

import threading
import weakref
from collections.abc import Iterable
from itertools import chain
from typing import TYPE_CHECKING

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)


class EarlyKVRecvReceiver:
    """Engine-side ZMQ PULL endpoint receiving worker early-recv reports."""

    def __init__(self) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)
        self._decoder = msgspec.msgpack.Decoder(list[str])
        self._closed = False
        self._finalizer = weakref.finalize(self, self._close_socket, self._socket)

    @staticmethod
    def _close_socket(sock: zmq.Socket) -> None:
        if not sock.closed:
            sock.close(linger=0)

    def drain(self) -> list[set[str]]:
        """Return pending reports; each message is one worker's vote."""
        if self._closed:
            return []
        notifications: list[set[str]] = []
        while True:
            try:
                message = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return notifications
            notifications.append(set(self._decoder.decode(message)))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._finalizer()


class EarlyKVRecvNotifier:
    """Worker-side thread-safe notifier pushing completed loads to the engine."""

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        self._lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, self._close_socket, self._socket)

    @staticmethod
    def _close_socket(sock: zmq.Socket) -> None:
        if not sock.closed:
            sock.close(linger=0)

    def try_notify(self, req_ids: Iterable[str]) -> bool:
        """Non-blocking send of completed receive request IDs.

        Returns True if the message was queued on the ZMQ socket, or False if
        the notifier is closed or the send would block/fail so the caller can
        fall back to ordinary step-output reporting without losing or
        duplicating a vote.
        """
        req_id_list = list(req_ids)
        if not req_id_list:
            return True
        payload = msgspec.msgpack.encode(req_id_list)
        with self._lock:
            if self._closed:
                return False
            try:
                self._socket.send(payload, flags=zmq.NOBLOCK)
                return True
            except zmq.ZMQError as e:
                logger.debug(
                    "Early KV receive notify fell back to step output: %s",
                    e,
                )
                return False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._finalizer()


class EarlyKVOutputAggregator(KVOutputAggregator):
    """Also counts receives that workers push between step outputs."""

    def __init__(
        self,
        expected_finished_count: int,
        receiver: EarlyKVRecvReceiver | None = None,
    ) -> None:
        super().__init__(expected_finished_count)
        self._receiver = receiver
        # Unfinished receives with a vote from a step output.
        self._step_output_recving = set[str]()

    @classmethod
    def from_connector(
        cls, connector: "KVConnectorBase", world_size: int
    ) -> "EarlyKVOutputAggregator":
        return cls(
            connector.get_finished_count() or world_size,
            receiver=EarlyKVRecvReceiver(),
        )

    @property
    def endpoint(self) -> str:
        assert self._receiver is not None
        return self._receiver.path

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        voted = {
            req_id
            for kv_output in chain(
                self._salvaged_outputs,
                (o.kv_connector_output for o in outputs if o is not None),
            )
            if kv_output is not None
            for req_id in kv_output.finished_recving or ()
        }
        output = super().aggregate(outputs, output_rank)
        if output is not None:
            self._step_output_recving = {
                req_id
                for req_id in self._step_output_recving | voted
                if req_id in self._recv_remaining_count
            }
        return output

    def add_finished_recving(self, req_ids: set[str]) -> set[str]:
        """Count one worker's report and return the requests now finished.

        Each worker reports a load once, so votes are not deduplicated.
        """
        deferred = req_ids & (self._failed_recving_pending | self._step_output_recving)
        if deferred:
            # Complete through a later step output, so the scheduler sees the
            # failure or invalid blocks first.
            self._defer_kv_connector_outputs(
                [KVConnectorOutput(finished_recving=deferred)]
            )
        finished = set[str]()
        self._count_finished(req_ids - deferred, self._recv_remaining_count, finished)
        return finished

    def take_early_finished_recving(self) -> set[str] | None:
        if self._receiver is None:
            return None
        finished: set[str] = set()
        for notification in self._receiver.drain():
            finished |= self.add_finished_recving(notification)
        return finished

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
