# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side client for peer prefix lookup-and-pin over NIXL.

The scheduler thread must never block on the network, so lookups run on one
background thread. The scheduler submits a lookup under a key, polls for the
result on later steps, and cancels it if the request goes away.

Every lookup gets its own REQ socket. The producer's replies are not FIFO (a
miss is answered at once while an earlier hit waits for the producer's step
fence) and a `LookupPinReply` carries no correlation id, so a socket shared
per peer could hand a reply to the wrong lookup. A REQ socket per in-flight
lookup correlates by construction and needs no wire change. The cost is one
TCP connect per lookup.
"""

import contextlib
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    LOOKUP_PIN_MSG,
    LookupPinReply,
    LookupPinRequest,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)

_POLL_MS = 50
_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class LookupResult:
    # None when the lookup failed.
    reply: LookupPinReply | None
    # "timeout", "bad_reply", "send_failed" or "closed" when reply is None.
    error: str | None = None


@dataclass
class _Job:
    job_id: int
    key: str
    host: str
    port: int
    frame: bytes


@dataclass
class _Active:
    job: _Job
    deadline: float


def _holds_pin(reply: LookupPinReply | None) -> bool:
    return reply is not None and bool(reply.remote_request_id)


class PeerLookupClient:
    """Asynchronous lookups against peers' side-channel ROUTER sockets.

    `submit`, `poll`, `cancel`, `take_orphaned_pins`, `has_in_flight`
    and `close` are for the scheduler thread and never block on the network.
    The background thread owns every socket and the context; the two sides
    share only the job queue, the wake pipe and the state guarded by `_lock`.
    """

    def __init__(
        self,
        timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._next_job_id = 0
        # key -> the live lookup, from submit until poll or cancel.
        self._live: dict[str, _Job] = {}
        # Finished lookups that poll has not taken; keys are a subset of _live.
        self._results: dict[str, LookupResult] = {}
        # Jobs cancelled while queued or in flight, until the thread notices.
        self._cancelled: set[int] = set()
        self._orphans: list[tuple[str, int, LookupPinReply]] = []
        self.num_lost_after_cancel = 0
        self._closed = False
        self._stop = threading.Event()
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)
        self._ctx = zmq.Context()
        self._thread = threading.Thread(
            target=self._run, name="nixl_peer_lookup", daemon=True
        )
        self._thread.start()

    # ---- scheduler thread -------------------------------------------------

    def submit(self, key: str, host: str, port: int, request: LookupPinRequest) -> None:
        """Start a lookup. A no-op while `key` is in flight or untaken."""
        frame = msgspec.msgpack.encode(
            (LOOKUP_PIN_MSG, msgspec.msgpack.encode(request))
        )
        with self._lock:
            if self._closed or key in self._live:
                return
            job = _Job(self._next_job_id, key, host, port, frame)
            self._next_job_id += 1
            self._live[key] = job
            self._jobs.put(job)
        self._wake()

    def poll(self, key: str) -> LookupResult | None:
        """The result once, then the key is forgotten; None while in flight."""
        with self._lock:
            result = self._results.pop(key, None)
            if result is not None:
                del self._live[key]
            return result

    def cancel(self, key: str) -> None:
        """The request went away; discard its lookup.

        A reply that is, or later turns out to be, a hit leaves a pin on the
        producer, which is recorded for `take_orphaned_pins`. A lookup whose
        socket is closed before its reply is counted in
        `num_lost_after_cancel`: the producer may hold a pin until its lease
        ends.
        """
        with self._lock:
            job = self._live.pop(key, None)
            if job is None:
                return
            result = self._results.pop(key, None)
            if result is None:
                self._cancelled.add(job.job_id)
            elif _holds_pin(result.reply):
                assert result.reply is not None
                self._orphans.append((job.host, job.port, result.reply))
        self._wake()

    def take_orphaned_pins(self) -> list[tuple[str, int, LookupPinReply]]:
        """Hits nobody will use: `(host, port, reply)` to release on the peer."""
        with self._lock:
            orphans, self._orphans = self._orphans, []
        return orphans

    def has_in_flight(self) -> bool:
        with self._lock:
            return len(self._live) > len(self._results)

    def close(self) -> None:
        """Stop the thread and release every socket and fd. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._wake()
        self._thread.join(_JOIN_TIMEOUT_S)
        with self._lock:
            for key in self._live:
                self._results.setdefault(key, LookupResult(None, "closed"))
        if self._thread.is_alive():
            # Closing under a running thread risks fd reuse and a blocked term.
            logger.warning("Peer lookup thread did not stop; leaking its resources")
            return
        self._ctx.destroy(linger=0)
        for fd in (self._wake_r, self._wake_w):
            with contextlib.suppress(OSError):
                os.close(fd)

    def _wake(self) -> None:
        # OSError: the pipe is full (a wake-up is pending) or already closed.
        with contextlib.suppress(OSError):
            os.write(self._wake_w, b"\x01")

    # ---- lookup thread ----------------------------------------------------

    def _complete(self, job: _Job, result: LookupResult) -> None:
        """Hand a finished lookup to poll, or record the pin if it was cancelled."""
        with self._lock:
            if job.job_id in self._cancelled:
                self._cancelled.discard(job.job_id)
                if _holds_pin(result.reply):
                    assert result.reply is not None
                    self._orphans.append((job.host, job.port, result.reply))
            elif self._live.get(job.key) is job:
                self._results[job.key] = result

    def _is_cancelled(self, job: _Job) -> bool:
        with self._lock:
            return job.job_id in self._cancelled

    def _start(self, job: _Job, active: dict[zmq.Socket, _Active], poller) -> None:
        with self._lock:
            if job.job_id in self._cancelled:
                self._cancelled.discard(job.job_id)  # Never sent: nothing to lose.
                return
        sock = None
        try:
            sock = make_zmq_socket(
                self._ctx,
                make_zmq_path("tcp", job.host, job.port),
                zmq.REQ,
                bind=False,
                linger=0,
            )
            sock.send(job.frame, flags=zmq.NOBLOCK)
            poller.register(sock, zmq.POLLIN)
            active[sock] = _Active(job, self._clock() + self._timeout_s)
        except Exception:
            logger.warning(
                "Peer lookup could not reach %s:%s", job.host, job.port, exc_info=True
            )
            if sock is not None:
                sock.close(linger=0)
            self._complete(job, LookupResult(None, "send_failed"))

    def _finish(self, sock, active, poller, result: LookupResult | None) -> None:
        """Close `sock`; deliver `result` unless it is None (cancelled)."""
        entry = active.pop(sock)
        with contextlib.suppress(Exception):
            poller.unregister(sock)
        with contextlib.suppress(Exception):
            sock.close(linger=0)
        if result is not None:
            self._complete(entry.job, result)

    def _run(self) -> None:
        active: dict[zmq.Socket, _Active] = {}
        poller = zmq.Poller()
        poller.register(self._wake_r, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                try:
                    self._step(active, poller)
                except Exception:
                    logger.warning("Peer lookup loop error", exc_info=True)
                    time.sleep(0.01)
        finally:
            for sock in list(active):
                self._finish(sock, active, poller, None)

    def _step(self, active: dict[zmq.Socket, _Active], poller) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            self._start(job, active, poller)

        # Cancelled in flight: take a reply that is already here (so a hit is
        # recorded as an orphan), else close and count the pin as lost.
        for sock, entry in list(active.items()):
            if not self._is_cancelled(entry.job):
                continue
            arrived: LookupResult | None = None
            with contextlib.suppress(Exception):
                if sock.poll(0, zmq.POLLIN):
                    arrived = self._read(sock)
            if arrived is not None:
                # _complete sees the cancel and records the orphan.
                self._finish(sock, active, poller, arrived)
                continue
            with self._lock:
                self._cancelled.discard(entry.job.job_id)
                self.num_lost_after_cancel += 1
            self._finish(sock, active, poller, None)

        try:
            events = dict(poller.poll(_POLL_MS))
        except zmq.ZMQError:
            if self._stop.is_set():
                return
            raise
        if self._wake_r in events:
            with contextlib.suppress(BlockingIOError, OSError):
                while os.read(self._wake_r, 4096):
                    pass

        now = self._clock()
        for sock, entry in list(active.items()):
            try:
                if sock in events:
                    result = self._read(sock)
                elif now >= entry.deadline:
                    result = LookupResult(None, "timeout")
                else:
                    continue
            except zmq.Again:
                continue
            except Exception:
                logger.warning("Peer lookup socket failed", exc_info=True)
                result = LookupResult(None, "bad_reply")
            self._finish(sock, active, poller, result)

    @staticmethod
    def _read(sock: zmq.Socket) -> LookupResult:
        frame = sock.recv(flags=zmq.NOBLOCK)
        try:
            reply = msgspec.msgpack.decode(frame, type=LookupPinReply)
        except (msgspec.DecodeError, msgspec.ValidationError):
            return LookupResult(None, "bad_reply")
        return LookupResult(reply)
