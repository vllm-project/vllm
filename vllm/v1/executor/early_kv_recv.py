# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report completed KV pulls between model outputs and the next schedule()."""

import os
import time

import torch
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.utils import sched_yield
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket

logger = init_logger(__name__)


class EarlyKVRecvReceiver:
    """Engine core end: the request ids whose KV loads finished, or the message
    of a failed poll."""

    def __init__(self) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)

    def drain(self) -> list[set[str] | str]:
        notifications: list[set[str] | str] = []
        while True:
            try:
                notifications.append(self._socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                return notifications

    def close(self) -> None:
        self._socket.close(linger=0)


# Interval between polls while the worker waits for its next RPC.
_POLL_INTERVAL_S = 0.0002
# Without a step RPC for this long the engine is idle or paused: stop polling
# and let the worker block in dequeue().
_ACTIVE_WINDOW_S = 1.0


class EarlyKVRecvSender:
    """Worker end: polls the KV connector between RPCs."""

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        self._last_activity = time.perf_counter()
        self._next_poll = 0.0

    def step_rpc_done(self) -> None:
        """Worker thread: an ``execute_model`` or ``sample_tokens`` RPC finished."""
        self._last_activity = time.perf_counter()

    def wait_for_poll(self, rpc_mq: MessageQueue) -> bool:
        """Spin until a poll is due while receives are pending.

        A poll is due every ``_POLL_INTERVAL_S`` while the worker waits for its
        next RPC: the engine may schedule at any point of that wait, and a
        receive it learns about in time runs one step earlier. Returns False
        once an RPC is readable, ``rpc_mq`` is shutting down, or no step RPC
        arrived for ``_ACTIVE_WINDOW_S``, so the worker blocks in ``dequeue()``.
        """
        if not has_kv_transfer_group():
            return False
        connector = get_kv_transfer_group()
        while connector.has_pending_recving() and not rpc_mq.shutting_down:
            if rpc_mq.readable():
                return False
            now = time.perf_counter()
            if now - self._last_activity > _ACTIVE_WINDOW_S:
                return False
            if now >= self._next_poll:
                return True
            sched_yield()
        return False

    def poll(self) -> None:
        """Send the receives that completed since the last poll.

        A failure is sent as its message, and the engine fails as it would on
        a failed step; raising it here instead would leave the other workers
        waiting in their next collective.
        """
        try:
            with torch.inference_mode():
                req_ids = get_kv_transfer_group().poll_finished_recving()
            if req_ids:
                self._socket.send_pyobj(req_ids)
        except Exception as e:
            logger.exception("Early KV receive poll failed")
            try:
                self._socket.send_pyobj(f"{type(e).__name__}: {e}", zmq.NOBLOCK)
            except Exception:
                logger.exception("Early KV receive error channel failed")
                # Wake the executor's worker monitor without entering collective
                # teardown or letting a utility RPC consume this fatal error.
                os._exit(1)
        finally:
            # Measured from the end of a poll, so long polls do not run back
            # to back.
            self._next_poll = time.perf_counter() + _POLL_INTERVAL_S

    def close(self) -> None:
        self._socket.close(linger=0)
