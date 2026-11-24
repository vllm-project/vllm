# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side client for peer prefix lookup-and-pin over NIXL.

The scheduler thread must never block on the network, so lookups run on one
background thread. The scheduler submits a lookup under a key, polls for the
result on later steps, and cancels it if the request goes away.

Capability probe. A `LOOKUP_PIN_MSG` frame sent to an engine that predates it
ends that engine's handshake listener (it decodes every frame as a 3-tuple
outside any try), which also breaks its ordinary P/D handshakes. So no lookup is
sent unprobed. The first lookup for a peer sends the one frame every engine
answers safely, `(GET_META_MSG, 0, 0)`, and reads `NixlHandshakePayload
.peer_lookup` from the reply (an engine that predates the field decodes as
False). Only a peer that advertises it gets the lookup; any other fails the
lookup with "peer_unsupported" and the request computes locally. A peer whose
engine id (from the payload's agent metadata) is not the one the router named
gets no lookup either ("peer_engine_mismatch"). The verdict is cached per peer
address and engine id (`cap_ttl_s` for a yes, `negative_ttl_s` for a no,
`failure_ttl_s` when the probe failed or the engine differed), and lookups that
arrive while a probe is out wait for it instead of probing again. The probe runs on the
background thread like everything else, so the scheduler never waits for it.

Every lookup gets its own REQ socket. The producer's replies are not FIFO (a
miss is answered at once while an earlier hit waits for the producer's step
fence), so a socket shared per peer could hand a reply to the wrong lookup. A REQ
socket per in-flight lookup correlates by construction, at the cost of one TCP
connect per lookup. `LookupPinRequest.lookup_id` is echoed in the reply and
checked here (a mismatch fails the lookup), so a later client could instead share
one DEALER socket per peer and route replies by that id.
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
    GET_META_MSG,
    LOOKUP_PIN_MSG,
    LookupPinReply,
    LookupPinRequest,
    NixlHandshakePayload,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)

_POLL_MS = 50
_JOIN_TIMEOUT_S = 5.0
# Any rank's handshake payload carries the engine-wide `peer_lookup` flag.
_PROBE_FRAME = msgspec.msgpack.encode((GET_META_MSG, 0, 0))


@dataclass(frozen=True)
class LookupResult:
    # None when the lookup failed.
    reply: LookupPinReply | None
    # When reply is None: "timeout", "bad_reply", "send_failed", "closed",
    # "lookup_id_mismatch", or one of the probe's, where no lookup was sent:
    # "peer_unsupported" (the peer does not serve lookups, or its payload did not
    # decode), "peer_engine_mismatch" (another engine answers at that address),
    # "probe_timeout", "probe_bad_reply" or "probe_send_failed".
    error: str | None = None


@dataclass
class _Job:
    job_id: int
    key: str
    host: str
    port: int
    frame: bytes
    # Identifies the peer's incarnation in the capability cache.
    engine_id: str | None = None
    # The id the reply must echo; None skips the check.
    lookup_id: str | None = None


class _EngineId(msgspec.Struct):
    """The one field of NixlAgentMetadata the probe reads."""

    engine_id: str


@dataclass
class _Active:
    job: _Job
    deadline: float
    # "probe" (GET_META for the capability check) or "lookup".
    kind: str = "lookup"


@dataclass
class _Capability:
    """What the probe learned about one peer; thread-private."""

    engine_id: str | None
    # None: serves lookups. Otherwise the error lookups to it fail with.
    error: str | None
    expires_at: float


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
        probe_timeout_s: float | None = None,
        cap_ttl_s: float = 60.0,
        negative_ttl_s: float = 30.0,
        failure_ttl_s: float = 5.0,
    ) -> None:
        self._timeout_s = timeout_s
        self._probe_timeout_s = (
            min(timeout_s, 2.0) if probe_timeout_s is None else probe_timeout_s
        )
        self._cap_ttl_s = cap_ttl_s
        self._negative_ttl_s = negative_ttl_s
        self._failure_ttl_s = failure_ttl_s
        self._clock = clock
        # Lookup thread only: (host, port) -> probe verdict, and the jobs
        # waiting on a probe that is out, per peer.
        self._caps: dict[tuple[str, int], _Capability] = {}
        self._probing: dict[tuple[str, int], list[_Job]] = {}
        self._warned_payload: set[tuple[str, int]] = set()
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

    @property
    def max_latency_s(self) -> float:
        """Longest a lookup takes to resolve: a probe, then the lookup."""
        return self._probe_timeout_s + self._timeout_s

    # ---- scheduler thread -------------------------------------------------

    def submit(
        self,
        key: str,
        host: str,
        port: int,
        request: LookupPinRequest,
        engine_id: str | None = None,
    ) -> None:
        """Start a lookup. A no-op while `key` is in flight or untaken.

        `engine_id` names the peer's incarnation: a verdict cached for another
        engine at the same address is not reused.
        """
        frame = msgspec.msgpack.encode(
            (LOOKUP_PIN_MSG, msgspec.msgpack.encode(request))
        )
        with self._lock:
            if self._closed or key in self._live:
                return
            job = _Job(
                self._next_job_id,
                key,
                host,
                port,
                frame,
                engine_id=engine_id,
                lookup_id=request.lookup_id,
            )
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
        """Begin `job`: probe the peer first unless its verdict is cached."""
        with self._lock:
            if job.job_id in self._cancelled:
                self._cancelled.discard(job.job_id)  # Never sent: nothing to lose.
                return
        cap = self._caps.get((job.host, job.port))
        if cap is not None and (
            cap.expires_at <= self._clock() or cap.engine_id != job.engine_id
        ):
            cap = None
        if cap is None:
            self._probe(job, active, poller)
        elif cap.error is not None:
            self._complete(job, LookupResult(None, cap.error))
        else:
            self._send_lookup(job, active, poller)

    def _open(
        self, job: _Job, frame: bytes, kind: str, timeout_s: float, active, poller
    ):
        """Send `frame` on a fresh REQ socket and track it; raises on failure."""
        sock = None
        try:
            sock = make_zmq_socket(
                self._ctx,
                make_zmq_path("tcp", job.host, job.port),
                zmq.REQ,
                bind=False,
                linger=0,
            )
            sock.send(frame, flags=zmq.NOBLOCK)
            poller.register(sock, zmq.POLLIN)
            active[sock] = _Active(job, self._clock() + timeout_s, kind)
        except Exception:
            if sock is not None:
                sock.close(linger=0)
            raise

    def _send_lookup(self, job: _Job, active, poller) -> None:
        try:
            self._open(job, job.frame, "lookup", self._timeout_s, active, poller)
        except Exception:
            logger.warning(
                "Peer lookup could not reach %s:%s", job.host, job.port, exc_info=True
            )
            self._complete(job, LookupResult(None, "send_failed"))

    def _probe(self, job: _Job, active, poller) -> None:
        addr = (job.host, job.port)
        if addr in self._probing:
            self._probing[addr].append(job)  # Wait for the probe already out.
            return
        self._probing[addr] = [job]
        try:
            self._open(
                job, _PROBE_FRAME, "probe", self._probe_timeout_s, active, poller
            )
        except Exception:
            logger.warning(
                "Peer lookup could not reach %s:%s", job.host, job.port, exc_info=True
            )
            self._resolve_probe(job, "probe_send_failed", active, poller)

    def _resolve_probe(self, job: _Job, error: str | None, active, poller) -> None:
        """Record the verdict for `job`'s peer and release the jobs waiting on it."""
        addr = (job.host, job.port)
        ttl = (
            self._cap_ttl_s
            if error is None
            else self._negative_ttl_s
            if error == "peer_unsupported"
            else self._failure_ttl_s
        )  # A mismatch or failed probe is retried soon.
        previous = self._caps.get(addr)
        self._caps[addr] = _Capability(job.engine_id, error, self._clock() + ttl)
        if previous is None or previous.error != error:
            if error is None:
                logger.info("Peer %s:%s serves prefix lookups", *addr)
            else:
                logger.info(
                    "Peer %s:%s will not get prefix lookups (%s); its requests "
                    "compute locally",
                    *addr,
                    error,
                )
        for waiting in self._probing.pop(addr, [job]):
            self._start(waiting, active, poller)

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
            if entry.kind != "lookup" or not self._is_cancelled(entry.job):
                continue
            arrived: LookupResult | None = None
            with contextlib.suppress(Exception):
                if sock.poll(0, zmq.POLLIN):
                    arrived = self._read(sock, entry.job)
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
            if sock not in active:  # Resolved by an earlier iteration.
                continue
            if entry.kind == "probe":
                self._step_probe(sock, entry, sock in events, now, active, poller)
                continue
            try:
                if sock in events:
                    result = self._read(sock, entry.job)
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

    def _step_probe(self, sock, entry: _Active, readable: bool, now, active, poller):
        error: str | None
        job = entry.job
        if readable:
            try:
                advertises, engine_id = self._read_probe(sock)
                if not advertises:
                    error = "peer_unsupported"
                elif job.engine_id is not None and engine_id != job.engine_id:
                    # A restart or downgrade at this address, or a router that
                    # is wrong: the engine here is not the one we were told.
                    logger.warning(
                        "Peer %s:%s answers as engine %r, expected %r; no lookup",
                        job.host,
                        job.port,
                        engine_id,
                        job.engine_id,
                    )
                    error = "peer_engine_mismatch"
                else:
                    error = None
            except zmq.Again:
                return
            except (msgspec.DecodeError, msgspec.ValidationError) as exc:
                # Not a payload this version can read: treat the peer as one
                # that does not serve lookups, and say so once per peer.
                addr = (job.host, job.port)
                if addr not in self._warned_payload:
                    self._warned_payload.add(addr)
                    logger.warning(
                        "Peer %s:%s sent a handshake payload that does not "
                        "decode (%s); it gets no lookups",
                        *addr,
                        exc,
                    )
                error = "peer_unsupported"
            except Exception as exc:
                logger.warning("Peer capability probe failed: %s", exc)
                error = "probe_bad_reply"
        elif now >= entry.deadline:
            error = "probe_timeout"
        else:
            return
        self._finish(sock, active, poller, None)
        self._resolve_probe(job, error, active, poller)

    @staticmethod
    def _read_probe(sock: zmq.Socket) -> tuple[bool, str | None]:
        """The handshake payload's `peer_lookup` and the replying engine's id.

        A payload from an engine that predates the field decodes as False. The
        reply has a second frame (the peer's clock) that is ignored. The id is
        None when the agent metadata cannot be read.
        """
        frames = sock.recv_multipart(flags=zmq.NOBLOCK)
        payload = msgspec.msgpack.decode(frames[0], type=NixlHandshakePayload)
        if not payload.peer_lookup:
            return False, None
        try:
            engine = msgspec.msgpack.decode(
                payload.agent_metadata_bytes, type=_EngineId
            )
        except (msgspec.DecodeError, msgspec.ValidationError):
            return True, None
        return True, engine.engine_id

    @staticmethod
    def _read(sock: zmq.Socket, job: _Job) -> LookupResult:
        frame = sock.recv(flags=zmq.NOBLOCK)
        try:
            reply = msgspec.msgpack.decode(frame, type=LookupPinReply)
        except (msgspec.DecodeError, msgspec.ValidationError):
            return LookupResult(None, "bad_reply")
        if job.lookup_id is not None and reply.lookup_id != job.lookup_id:
            # Cannot happen with a socket per lookup unless the peer is broken.
            # The reply is not ours, so any pin it names is not ours to free.
            logger.warning(
                "Peer lookup reply from %s:%s answers lookup %r, expected %r; "
                "dropping it",
                job.host,
                job.port,
                reply.lookup_id,
                job.lookup_id,
            )
            return LookupResult(None, "lookup_id_mismatch")
        return LookupResult(reply)
