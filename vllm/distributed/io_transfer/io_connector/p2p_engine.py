# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2pEngine — ZMQ-based byte-transfer engine for IO connector.

Transfers arbitrary bytes between disaggregated vLLM roles using a
ZMQ ROUTER/DEALER pattern.

Protocol (GET mode)
-------------------
The sender buffers objects locally. The receiver pulls them on demand.

  put_object(object_id, frames)          — stores a list of buffers in send_store
                                           (non-blocking)
  fetch_object(object_id, sender_addr)   — receiver sends GET, awaits reply without
                                           blocking the calling thread or the main
                                           event loop; returns list[bytes]
  delete_object(object_id)               — sender discards a buffered object

Wire protocol for fetch_object:
  Receiver DEALER → Sender ROUTER  :
    frame 0: msgpack {"cmd": "GET", "object_id": <id>, "correlation_id": <int>}

  Sender ROUTER → Receiver DEALER (success):
    frame 0: msgpack {"ok": True,  "correlation_id": <int>, "num_frames": N}
    frame 1..N: raw payload buffers (each sent with copy=False)

  Sender ROUTER → Receiver DEALER (error):
    frame 0: msgpack {"ok": False, "error": <str>, "correlation_id": <int>}

Each payload buffer is a separate ZMQ frame.  The envelope (frame 0) stays
small and cheap to encode; the payload frames are passed to
``send_multipart`` with ``zmq.Frame(..., copy=False)`` so ZMQ references the
backing buffers directly without an additional memcpy.

Correlation ID
--------------
Every fetch_object call generates a unique ``correlation_id`` from a
per-connection monotonic integer counter.  The ID is sent in the request
(as a msgpack integer — more compact than a string) and echoed back in the
reply.  A persistent reader loop on the async event loop receives all
incoming replies on the shared DEALER socket and dispatches each one to the
waiting asyncio.Future via the pending dict keyed by ``correlation_id``.

Socket pool
-----------
A single persistent ``zmq.asyncio`` DEALER socket is kept open per
``remote_address``.  All concurrent ``fetch_object`` calls to the same
remote are multiplexed over that one socket without any per-request
connection setup (no TCP handshake, no ZMTP negotiation after the first
call). Sockets are created lazily on first use and reused for the
lifetime of the engine.

Non-blocking design
-------------------
``fetch_object`` is synchronous on the outside (the ``IOConnectorBase``
interface does not require async).  Internally the ZMQ send/recv uses
``zmq.asyncio`` DEALER sockets so the OS thread is never blocked on I/O.

To bridge the sync interface with the async implementation, P2pEngine owns a
private asyncio event loop that runs permanently in a dedicated daemon thread
(``_async_loop_thread``).  Each ``fetch_object`` call dispatches
``_fetch_object_async`` onto that loop via
``asyncio.run_coroutine_threadsafe`` and blocks only on the returned
``concurrent.futures.Future``, not on the ZMQ socket.

Single async context
--------------------
Both the server-side ROUTER and the client-side DEALER pool share the same
``zmq.asyncio.Context`` and the same private event loop.  A single daemon
thread runs the loop; no separate listener thread is needed.  The ROUTER
receive loop (``_serve``) and each connection's reply reader
(``_reply_reader``) are all asyncio Tasks on that loop, so the OS scheduler
never has to wake more than one thread to handle I/O.

Performance notes
-----------------
* ZMQ context is created with ``io_threads=2`` so the ZMQ I/O thread pool
  can keep up with high-bandwidth tensor transfers.  Tune upward for
  aggregate throughputs above ~10 Gb/s.
* A single ``msgspec.msgpack.Encoder`` instance is reused for all encode calls,
  avoiding per-call temporary dict allocations.
* ``_serve`` processes each GET request inline (no ``create_task`` per
  request) and attempts a ``NOBLOCK`` send first; only falling back to the
  async send when the kernel send buffer is momentarily full.
* ``_send_store`` is mutated exclusively on the async event-loop thread
  (all external writers route through ``run_coroutine_threadsafe``), so no
  ``threading.Lock`` is needed.
* Correlation IDs are plain integers throughout (msgpack int is smaller
  than msgpack str); the pending dict uses ``int`` keys.
* ``asyncio.Future`` is resolved directly with ``list[bytes] | None``
  rather than an intermediate ``(envelope, payload_frames)`` tuple.
"""

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass, field

import msgspec
import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)


@dataclass
class _StoreEntry:
    """Mutable store entry so refcount can be updated without allocating a new tuple."""

    frames: list["bytes | memoryview"]
    zmq_frames: list[zmq.Frame]
    num_frames: int
    refcount: int


@dataclass
class _RemoteConn:
    """State for one persistent DEALER connection to a remote address."""

    sock: zmq.asyncio.Socket
    # Maps correlation_id (int) → Future resolved with list[zmq.Frame] | None.
    # None means the server reported object-not-found.
    pending: dict[int, "asyncio.Future[list[zmq.Frame] | None]"] = field(
        default_factory=dict
    )
    # Monotonically increasing counter used to generate correlation IDs.
    # Incremented only from coroutines on the single-threaded _async_loop,
    # so no atomics are needed.  Unique within the lifetime of this connection.
    next_id: int = 0
    # The reader task drains replies from ``sock`` and resolves futures.
    reader_task: "asyncio.Task | None" = None


class P2pEngine:
    """Byte-transfer engine backed by ZMQ ROUTER/DEALER sockets.

    Server side
        An asyncio Task (``_serve``) runs a ``zmq.asyncio`` ROUTER that
        serves GET requests from remote receivers inline — no per-request
        Task is spawned, eliminating per-request asyncio.Task and
        asyncio.Future allocations. The send is attempted with NOBLOCK
        first; a true async fallback is used only when the kernel buffer is
        full (rare under normal operation).

    Client side
        A persistent ``zmq.asyncio`` DEALER socket is maintained per remote
        address in ``_conn_pool``.  A per-connection reader task continuously
        drains replies and dispatches them to the correct waiting
        ``asyncio.Future`` (keyed by integer correlation_id) via ``pending``.
        No per-request socket creation, TCP handshake, or ZMTP negotiation
        after the first call to a given remote.

    Store ownership
        ``_send_store`` is only ever mutated from coroutines running on
        ``_async_loop``.  External callers (``put_object``, ``delete_object``)
        route their mutations through ``run_coroutine_threadsafe`` so no
        ``threading.Lock`` is required.
    """

    def __init__(self, bind_host: str, bind_port: int) -> None:
        self._host = bind_host
        self._port = bind_port
        self.zmq_address = f"{self._host}:{self._port}"

        # io_threads=2: one thread per ~5 Gb/s of aggregate ZMQ throughput is
        # a reasonable starting point; increase further for very high bandwidth.
        self._async_context = zmq.asyncio.Context(io_threads=2)

        # Server-side ROUTER — created here (thread-safe) and bound; the
        # actual recv/send calls happen inside _serve on the event loop.
        self._router: zmq.asyncio.Socket = self._async_context.socket(zmq.ROUTER)
        self._router.bind(f"tcp://{self.zmq_address}")

        # Persistent DEALER socket pool: remote_address → _RemoteConn.
        # Accessed only from coroutines on _async_loop — no lock needed.
        self._conn_pool: dict[str, _RemoteConn] = {}

        # Send buffer: object_id → _StoreEntry.
        # Written by caller threads (put_object, delete_object) and read/mutated
        # by _serve on the event loop.  The lock is held only for the duration of
        # a single dict operation — no I/O, no awaiting — so contention is negligible.
        # _StoreEntry.refcount is updated in-place; no tuple is allocated per change.
        self._send_store: dict[str, _StoreEntry] = {}
        self._send_store_lock = threading.Lock()

        # Reusable encoder for the small control envelopes.
        self._packer = msgspec.msgpack.Encoder()

        # Start the event loop in a daemon thread, then schedule _serve as a
        # Task so the ROUTER is ready before any fetch_object call arrives.
        self._async_loop = asyncio.new_event_loop()
        self._async_loop_thread = threading.Thread(
            target=self._async_loop.run_forever,
            daemon=True,
            name="p2p-engine-async-loop",
        )
        self._async_loop_thread.start()

        # Schedule the server coroutine and wait until it has started so
        # that the ROUTER is actually polling before __init__ returns.
        started = threading.Event()
        asyncio.run_coroutine_threadsafe(self._serve(started), self._async_loop)
        started.wait()

        logger.info("P2pEngine init: zmq_address=%s", self.zmq_address)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def put_object(self, object_id: str, frames: list[bytes | memoryview]) -> None:
        """Buffer *frames* under *object_id* for a future fetch_object call.

        *frames* is a list of raw buffers that will be sent as individual ZMQ
        frames on the wire.  The caller is responsible for keeping references
        to any memoryview objects alive until the object is fetched or deleted.

        If *object_id* is already present the refcount is incremented by one
        and the frames are left unchanged. This handles the case where a sender
        registers the same object twice, meaning that this object will be
        fetched twice.

        The lock is held only for the duration of the dict mutation (nanoseconds);
        no I/O is performed, so there is no meaningful contention cost.
        """
        with self._send_store_lock:
            existing = self._send_store.get(object_id)
            if existing is not None:
                existing.refcount += 1  # mutate in-place — no tuple allocation
                logger.debug(
                    "P2pEngine: object_id=%s already buffered, refcount→%d",
                    object_id,
                    existing.refcount,
                )
            else:
                self._send_store[object_id] = _StoreEntry(
                    frames=frames,
                    zmq_frames=[zmq.Frame(f, copy=False) for f in frames],
                    num_frames=len(frames),
                    refcount=1,
                )

                logger.debug(
                    "P2pEngine: buffered object_id=%s "
                    "(%d frames, %d bytes total, refcount=1)",
                    object_id,
                    len(frames),
                    sum(len(f) for f in frames),
                )

    def fetch_object(
        self, object_id: str, remote_address: str
    ) -> list[zmq.Frame] | None:
        """Pull *object_id* from *remote_address*, blocking the calling thread.

        Returns a list of raw byte buffers (one per frame sent by the server),
        or ``None`` if the object was not found.

        Prefer ``fetch_object_future`` when calling from an async context: it
        returns a ``concurrent.futures.Future`` that can be wrapped with
        ``asyncio.wrap_future`` so no OS thread is blocked during the RTT.
        """
        return self.fetch_object_future(object_id, remote_address).result()

    def fetch_object_future(
        self, object_id: str, remote_address: str
    ) -> "concurrent.futures.Future[list[zmq.Frame] | None]":
        """Schedule a fetch and return the underlying concurrent.futures.Future.

        The future resolves when the ZMQ reply arrives on ``_async_loop``.
        Callers in an async context should wrap this with
        ``asyncio.wrap_future(engine.fetch_object_future(...))`` to await the
        result without blocking any OS thread.
        """
        return asyncio.run_coroutine_threadsafe(
            self._fetch_object_async(object_id, remote_address),
            self._async_loop,
        )

    async def _get_conn(self, remote_address: str) -> _RemoteConn:
        """Return (creating if needed) the persistent connection to *remote_address*.

        Must only be called from coroutines running on ``_async_loop``.
        No locking is required because the async loop is single-threaded.
        """
        conn = self._conn_pool.get(remote_address)
        if conn is not None:
            return conn

        sock = self._async_context.socket(zmq.DEALER)
        # Use a stable identity derived from our address so the ROUTER can
        # route unsolicited messages back to us if needed in future.
        sock.setsockopt_string(zmq.IDENTITY, f"{self.zmq_address}/dealer")
        sock.connect(f"tcp://{remote_address}")

        conn = _RemoteConn(sock=sock)
        conn.reader_task = self._async_loop.create_task(
            self._reply_reader(remote_address, conn),
            name=f"p2p-reader-{remote_address}",
        )
        self._conn_pool[remote_address] = conn
        logger.debug("P2pEngine: opened persistent DEALER to %s", remote_address)
        return conn

    async def _reply_reader(self, remote_address: str, conn: _RemoteConn) -> None:
        """Drain replies from *conn.sock* and resolve waiting futures.

        Runs as a long-lived asyncio Task for the lifetime of the connection.
        Each reply is: envelope frame (frame 0) + N payload frames (frames 1..N)
        on success, or envelope only on error.  The reader resolves the waiting
        Future directly with ``list[bytes]`` on success or ``None`` on error,
        avoiding an intermediate tuple allocation.
        If the socket is closed during shutdown the recv raises ZMQError and
        this task exits cleanly.
        """
        sock = conn.sock
        while True:
            try:
                frames = await sock.recv_multipart(copy=False)
            except zmq.ZMQError:
                # Socket closed — exit the reader loop.
                break

            if not frames:
                logger.warning(
                    "P2pEngine: empty multipart reply from %s, dropping",
                    remote_address,
                )
                continue

            try:
                envelope: dict = msgspec.msgpack.decode(frames[0].buffer)
            except Exception:
                logger.exception(
                    "P2pEngine: reply_reader failed to decode envelope from %s",
                    remote_address,
                )
                continue

            # correlation_id is an int on the wire (msgpack integer).
            correlation_id: int | None = envelope.get("correlation_id")
            if correlation_id is None:
                logger.warning(
                    "P2pEngine: reply from %s has no correlation_id, dropping",
                    remote_address,
                )
                continue

            fut = conn.pending.pop(correlation_id, None)
            if fut is None:
                logger.warning(
                    "P2pEngine: no pending request for correlation_id=%r "
                    "from %s, dropping",
                    correlation_id,
                    remote_address,
                )
                continue

            if fut.done():
                continue

            if envelope.get("ok"):
                # Resolve directly with the payload frames — no intermediate tuple.
                payload_frames: list[zmq.Frame] = frames[1:]
                logger.debug(
                    "P2pEngine: fetched object (%d frames, %d bytes total) from %s",
                    len(payload_frames),
                    sum(len(f) for f in payload_frames),
                    remote_address,
                )
                fut.set_result(payload_frames)
            else:
                logger.error(
                    "P2pEngine: fetch failed from %s: %s",
                    remote_address,
                    envelope.get("error"),
                )
                fut.set_result(None)

    async def _fetch_object_async(
        self, object_id: str, remote_address: str
    ) -> list[zmq.Frame] | None:
        """Async core of fetch_object; runs on ``_async_loop``.

        Reuses the persistent DEALER socket for *remote_address* from the
        connection pool.  A unique integer ``correlation_id`` is generated
        per call from a per-connection monotonic counter (no OS calls); the
        reply reader task resolves the corresponding ``asyncio.Future``
        directly with ``list[bytes] | None`` when the matching reply arrives.

        Returns a list of raw byte buffers (one per sent frame) on success,
        or ``None`` if the server reports the object was not found.
        """
        conn = await self._get_conn(remote_address)
        correlation_id = conn.next_id
        conn.next_id += 1

        fut: asyncio.Future[list[zmq.Frame] | None] = self._async_loop.create_future()
        conn.pending[correlation_id] = fut

        request = self._packer.encode(
            {
                "cmd": "GET",
                "object_id": object_id,
                "correlation_id": correlation_id,
            }
        )

        logger.debug(
            "GET object_id=%r from remote_address=%r", object_id, remote_address
        )
        try:
            # Attempt a non-blocking send first; fall back to the async send
            # only when the kernel buffer is momentarily full (rare).
            try:
                conn.sock.send(request, flags=zmq.NOBLOCK)
            except zmq.Again:
                await conn.sock.send(request)
        except Exception:
            conn.pending.pop(correlation_id, None)
            raise

        result = await fut
        logger.debug(
            "RECV object_id=%r from remote_address=%r", object_id, remote_address
        )
        return result

    def delete_object(self, object_id: str) -> None:
        """Discard a buffered object that was never fetched.

        Serving a GET request decrements the refcount and removes the entry
        only when it reaches zero, so this is a no-op for objects that were
        fully consumed. It only has real work to do when the receiver abandons
        the object without ever calling fetch_object (e.g. request cancelled
        before get_data).
        """
        with self._send_store_lock:
            self._send_store.pop(object_id, None)

    def shutdown(self) -> None:
        """Close all ZMQ sockets and terminate the async context."""

        # Cancel the server task and all pooled DEALER sockets from inside
        # the loop so every outstanding await sees the cancellation before
        # the context is torn down.
        async def _close_all() -> None:
            self._router.close(linger=0)
            for conn in self._conn_pool.values():
                conn.sock.close(linger=0)
                if conn.reader_task is not None:
                    conn.reader_task.cancel()
            self._conn_pool.clear()

        future = asyncio.run_coroutine_threadsafe(_close_all(), self._async_loop)
        future.result(timeout=2)

        self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        self._async_loop_thread.join(timeout=2)
        self._async_loop.close()
        self._async_context.term()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _serve(self, started: threading.Event) -> None:
        """Server coroutine: accept GET requests and handle each one inline.

        Runs as a long-lived asyncio Task on ``_async_loop``.  Uses the
        ``zmq.asyncio`` ROUTER so recv_multipart never blocks the loop.
        Signals *started* once the first iteration is about to begin so
        that ``__init__`` can return only after the ROUTER is ready.

        Each GET is handled inline rather than spawning a new asyncio.Task,
        eliminating per-request Task + Future + frame-list heap allocations.
        The reply send is attempted with NOBLOCK first; the async send path
        is used only when the kernel buffer is transiently full.
        """
        packer = self._packer
        router = self._router
        send_store = self._send_store

        started.set()
        while True:
            try:
                requester_id, message = await router.recv_multipart()
            except zmq.ZMQError:
                # Socket closed during shutdown — exit cleanly.
                break

            # --- decode request ---
            try:
                data = msgspec.msgpack.decode(message)
            except Exception:
                logger.exception("P2pEngine: failed to decode message")
                continue

            logger.debug("GET data=%r from requester_id=%r", data, requester_id)

            if data.get("cmd") != "GET":  # type: ignore[assignment]
                logger.warning(
                    "P2pEngine: unexpected command from %s: %r",
                    requester_id.decode(errors="replace"),
                    data.get("cmd"),  # type: ignore[assignment]
                )
                continue

            object_id: str = data["object_id"]  # type: ignore[assignment]
            correlation_id = data["correlation_id"]  # type: ignore[assignment]

            # --- look up payload ---
            with self._send_store_lock:
                entry = send_store.get(object_id)
                if entry is not None:
                    entry.refcount -= 1  # mutate in-place — no tuple allocation
                    if entry.refcount <= 0:
                        del send_store[object_id]

            # --- build reply frames ---
            frames: list[bytes | zmq.Frame]
            if entry is not None:
                envelope = packer.encode(
                    {
                        "ok": True,
                        "correlation_id": correlation_id,
                        "num_frames": entry.num_frames,
                    }
                )

                frames = [requester_id, envelope]
                frames.extend(entry.zmq_frames)

                logger.debug(
                    "P2pEngine: serving object_id=%s (%d frames, %d bytes) to %s",
                    object_id,
                    entry.num_frames,
                    sum(len(f) for f in entry.frames),
                    requester_id.decode(errors="replace"),
                )
            else:
                envelope = packer.encode(
                    {
                        "ok": False,
                        "error": f"object_id {object_id!r} not found",
                        "correlation_id": correlation_id,
                    }
                )
                frames = [requester_id, envelope]
                logger.warning(
                    "P2pEngine: GET for unknown object_id=%s from %s",
                    object_id,
                    requester_id.decode(errors="replace"),
                )

            # --- send reply (NOBLOCK fast path, async fallback) ---
            try:
                try:
                    router.send_multipart(frames, flags=zmq.NOBLOCK)
                except zmq.Again:
                    await router.send_multipart(frames)
            except zmq.ZMQError:
                continue

            logger.debug("SEND reply to requester_id=%r", requester_id)
